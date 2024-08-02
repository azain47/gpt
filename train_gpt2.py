# import model
from gpt2 import GPT2, GPTConfig
# import tokenizer
import tiktoken
# misc
import torch
import math, os, time
# import matplotlib.pyplot as plt

# Dataset Loader
class DataLoader():
    def __init__(self, B, T, file):
        self.B = B
        self.T = T
        
        with open(os.path.join(file)) as f:
            text = f.read()
        
        # tokenizer 
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.num_batches = len(self.tokens)//(B*T)

        print(f'Total tokens:{len(self.tokens)}')
        print(f'Total no. of batches : {self.num_batches}')
        # pointer to manage batches
        self.curr = 0    
    
    def get_next_batch(self):
        
        buf = self.tokens[self.curr : self.curr + (self.B * self.T) + 1]

        x = buf[:-1].view(self.B,self.T)
        y = buf[1:].view(self.B,self.T)

        self.curr += self.B * self.T

        if (self.curr + (self.B * self.T) + 1) > len(self.tokens):
            self.curr = 0

        return x, y 

# HyperParams
total_batch_size = 32768   # in number of tokens  
mini_batch_size = 6
context_length = 1024
# gradient accumulation, reach a desired batch size, by accum gradients over a smaller batch size.
grad_accum_steps = total_batch_size // (mini_batch_size * context_length)
print(f'Desired Batch size (in tokens): {total_batch_size}\nAccum Steps Required:{grad_accum_steps}')

# gpt2 hyperparams
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 120 
# 8000 * 32768 = 262 million tokens (overfitting, or grokking ;) )
max_steps = 8000

device = 'cpu'
if torch.cuda.is_available():
    device = "cuda"
print(f'Using device: {device}')

# Model Params
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embed=768
)
# Model Init, using the config.
model = GPT2(config)
model = model.to(device)

print(f'Model Size : {(sum(p.numel() for p in model.parameters())/1e6):.1f} M params')
# reducing matmul precision from float32 to tf32.
torch.set_float32_matmul_precision('high')

# Dataset Load
train_loader = DataLoader(mini_batch_size, context_length, 'shakespeare.txt')

# GPT2 LR Schedule, Linear Warmup followed by cosine decay.
def get_lr(iter):
    if iter < warmup_steps:
        # use linearLR
        return max_lr * (iter+1) / warmup_steps
    
    if iter > max_steps:
        return max_lr
    
    # use cosine decay for rest
    decay = (iter - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 + (1.0 + math.cos(math.pi * decay))
    
    # adding min_lr to not let the lr reach zero.
    return min_lr + coeff * (max_lr - min_lr)

optimizer = torch.optim.AdamW(model.parameters(), lr = max_lr, betas=(0.9,0.95))

# Finally, Train the model.

losses = []
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0
    for mini_step in range(grad_accum_steps):
        x, y = train_loader.get_next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # compensate for accumulation due to 'mean' reduction in the loss function.
        loss = loss / grad_accum_steps
        # compute gradients
        loss_accum+= loss.detach()
        loss.backward()
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)
    tokens_processed_ps = train_loader.B * train_loader.T * grad_accum_steps / dt
    losses.append(loss_accum)
    print(f"step{step+1} | loss:{loss_accum.item():.5f} | lr: {lr:.3e} | t:{dt*1000:.2f}ms | norm: {norm.detach():.3f} | tok/s: {tokens_processed_ps:.2f}")

# save model

torch.save(model.state_dict(), 'GPT2-Shakespeare.pt')
