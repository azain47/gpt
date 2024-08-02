import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 256 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' 
eval_iters = 200
n_embed = 384   # embedding dimensions
n_layer = 6     # number of transformer blocks
n_heads = 8     # number of heads in multi head attention
dropout = 0.2
# ------------

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoder decoder / tokenizer 
stoi = {ch:i for i,ch in enumerate(chars)} # simple mapping from set of unique chars
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]  # encode function to lookup the stoi dict
decode = lambda i: ''.join([itos[x] for x in i])  #decode func to lookup the itos dict

# dataset creation
data = torch.tensor(encode(text), dtype=torch.long)

# Also split the dataset into train-val
n = int(0.9 * len(data))  # 90-10 split
train_data = data[:n]
val_data = data[n:]

# Batching of dataset
# The dataset is divided on basis of 2 dimensions. Time Dimension and Batch Size.
# Time dimension is basically the context length. How many tokens our model sees before it predicts the next token.
# Karpathy says it as block_size.
# The Batch Dimension or Batch size is just no. of independent sequences the model will train on parallely(right word?)
# basically the batch size in a mini batch gradient descent, nothing else.

def get_batch(split):
    # generate a mini batch of inputs, targets
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - batch_size, (batch_size,))
    x = torch.stack([data[i:block_size+i] for i in ix]).to(device)
    y = torch.stack([data[i+1:block_size+i+1] for i in ix]).to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """single head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self,x):
        B,T,C = x.shape
        # (B,T,head_size)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # compute attention scores
        weights = q @ k.transpose(-2,-1) * self.head_size ** -0.5   # B,T,head_size * B,head_size,T -> B,T,T
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        # perform weighted aggregation
        out = weights @ v    # B,T,T * B,T,head_size -> B,T,head_size
        return out

class MultiHeadAttention(nn.Module):
    """multiple self attention heads, running in parallel."""
    
    def __init__(self, num_heads, head_size ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # concat in channel dimension
        out = self.proj(out)
        out = self.dropout(out)

        return out
        
class FeedForward(nn.Module):
    """simple MLP, adding depth post-attention"""
    
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """transformer block, communication (Attenshun) followed by computation"""

    def __init__(self, num_heads,n_embed) -> None:
        super().__init__()
        # example: 4 heads of 8 dimensions concat in a single dimension to give us 32 dimensions, i,e n_embed.
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.net = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.net(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # directly reads off the probability for next token 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_heads,n_embed) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        positional_embeddings = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        # add both embeddings
        x = token_embeddings + positional_embeddings
        # apply transformer blocks
        x = self.blocks(x)
        # final layer norm
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last tokens of block size.
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = GPT().to(device)


# Training the model
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))