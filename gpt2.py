from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

import torch.utils

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed, 2 * config.n_embed)
        self.c_attn_q = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.RESIDUAL_SCALING_INIT = 1
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.shape

        kv = self.c_attn(x)
        q = self.c_attn_q(x)

        k,v = kv.split(self.n_embed, dim = 2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

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

        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_head)]),
            ln_f = nn.LayerNorm(config.n_embed)
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

        pos = torch.arange(0, T , dtype=torch.long, device=idx.device)

        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return logits


# # Device setting
# device = 'cpu'
# if torch.cuda.is_available():
#     device = "cuda"
# print(f'Using device: {device}')

# # Model Init 
# model = GPT(GPTConfig(vocab_size=50304))
# # doesnt work on windows :(
# # model = torch.compile(model)
# model.to(device)


# # Training

# # import sys; sys.exit()
# tokens = enc.encode('Hello how are you!')
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0)
# x = tokens.to(device)

# max_length = 100

# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = model(x)

#         logits = logits[:,-1,:]

#         probs = F.softmax(logits, dim=-1)
        
#         topk_probs, topk_indices = torch.topk(probs,50)

#         ix = torch.multinomial(topk_probs, 1)

#         xcol = torch.gather(topk_indices, -1, ix)

#         x = torch.cat((x,xcol), dim=1)

# tokens = x[0].tolist()

# decoded = enc.decode(tokens)
# print(f'>:{decoded}')
