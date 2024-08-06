from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.utils
import numpy as np

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 16
    n_head: int = 16
    n_embed: int = 768
    use_rotary:bool = True
    
# Root Mean Square Normalisation, an alternate to LayerNorm.
# The only difference between RMSNorm and LayerNorm is the lack of recentering of data i.e. subtraction of mean.
# rest remains mostly the same, just 1 parameter, gamma.
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.d = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    # def _norm(self, x):
    #     # rqsrt = 1/sqrt
    #     return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # taken from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
        norm_x = x.norm(2, dim=-1, keepdim=True)

        rms_x = norm_x * self.d ** (-0.5)
        x_normed = x / (rms_x  + self.eps)
        
        return self.gamma * x_normed
    
# Rotary Position Embeddings
# basically view the query and key embeddings as vectors, and use a rotation matrix to rotate the vectors according
# to their position. this rotational change allows tokens to have relative distance between them, tokens farther away 
# will be rotated more, tokens closer will be rotated less.
def precompute_theta_freqs(dim, context_length):
    assert dim % 2 == 0, "Dimension must be divisible by 2!"
    # pre compute the theta frequencies of cosine and sine matrix using the formula in the paper.
    # theta = 10000 ^ (-2(i-1)/d) , where d = {1...dim/2}
    # NOTE: dim here refers to head_dim, rather than embed_dim. since we will be dealing with multi-head attn
    # hence, the q, k will be split in multiple heads of diff dimensions, which will concat to become embed_dim.
    thetas = (1.0 / (10000 ** (torch.arange(0, dim,2) / dim))).float()
    m = torch.arange(context_length)
    # product of every position with every theta.
    m_theta = torch.outer(m, thetas).float()
    # (seqlen, dim / 2)
    cos_freqs = torch.cos(m_theta).float()
    sin_freqs = torch.sin(m_theta).float()
    # (seqlen, dim)
    cos_freqs = np.repeat(cos_freqs,repeats=2, axis=-1)
    sin_freqs = np.repeat(sin_freqs,repeats=2, axis=-1)

    return cos_freqs, sin_freqs

# this gives us alternating x, as requirement in paper
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2] 
    x2 = x[..., x.shape[-1] // 2 :] 
    return torch.cat((-x2, x1), dim=-1) 

def apply_rotary_embeddings(q, k, freqs_cos, freqs_sin):
    # cos and sine freqs are of (seq_len, dim), so in our case (1024,64).
    # our query and key vectors are (B, nh, T, hs) wher nh = 12, hs = 768/12 = 64
    # (B, 12, T, 64) * (1, 1, T, 64) = (B, 12, 1024, 64)
    T = q.shape[2]
    # trim freq to context length as per input.
    # during training all inputs will have seqlen = maxseqlen but during inferencing that wont be the case.
    # unsqueeze will add 1 dimension to the start. (seqlen, dim) -> (1, seqlen, dim), rest 
    # broadcasting will take care
    freqs_cos = freqs_cos[:T].unsqueeze(0).to(q.device)
    freqs_sin = freqs_sin[:T].unsqueeze(0).to(q.device)
    q = (q * freqs_cos) + (rotate_half(q) * freqs_sin)
    k = (k * freqs_cos) + (rotate_half(k) * freqs_sin)
    
    return q, k    
        
class CausalSelfAttention(nn.Module):
    def __init__(self, config:GPTConfig) -> None:
        super().__init__()

        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed, 2 * config.n_embed)
        self.c_attn_v = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.RESIDUAL_SCALING_INIT = 1
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_head
        self.use_rotary = config.use_rotary
        # rotary positional embeddings
        if self.use_rotary:
            self.cos_freqs, self.sin_freqs = precompute_theta_freqs(self.head_dim, config.block_size)

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.shape

        qk = self.c_attn(x)
        v = self.c_attn_v(x)

        q,k = qk.split(self.n_embed, dim = 2)
        # before transpose, q,k are (B,T,nh,hs) where hs = embed / nh. after transpose, (B,nh,T,hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_rotary:
            q, k = apply_rotary_embeddings(q, k, self.cos_freqs, self.sin_freqs)
        # att = (q @ k.transpose(-1,-2) * (1.0/math.sqrt(k.size(-1))))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # att = F.softmax(att, dim=-1)

        # y = att @ v
        # replace attn calculations with flash attention. Reduced the time from ~12,000 ms to ~1200. crazy. 

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C)

        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.RESIDUAL_SCALING_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.ln_1 = RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config:GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embed)
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # weight sharing scheme
        # self.transformer.wte.weight = self.lm_head.weight
        # custom weight init
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            std = 0.02
            
            if hasattr(module, 'RESIDUAL_SCALING_INIT'):
                std += (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        B, T = idx.size()

        assert T <= self.config.block_size
        # Token Embeddings
        x = self.transformer.wte(idx)

        # Add support for rotary embeddings
        if self.config.use_rotary == False:
            # empirical pos embeddings
            pos = torch.arange(0, T , dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = x + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return logits
