from gpt2 import GPT2, GPTConfig

import torch, time, math, os
import pickle
import random
import matplotlib.pyplot as plt
from typing import Dict, Optional, Literal

# Dataset Loader
class DataLoader():
    def __init__(self, B, T, split):
        self.B = B
        self.T = T      
        
        datasets = os.listdir('./dataset')
        file = [s for s in datasets if split in s] 
        # tokens = read_file(os.path.join('./dataset/', file[0]))
        with open((os.path.join('./dataset/', file[0])), 'rb') as f:
            self.documents = pickle.load(f)
        
        # self.tokens = tokens
        self.shuffle_docs()
        self.flatten_docs()

        self.num_batches = len(self.tokens)//(B*T)

        print(f'Total tokens:{len(self.tokens)}')
        print(f'Total no. of batches : {self.num_batches}')
        # pointer to manage batches
        self.curr = 0    
    
    def shuffle_docs(self):
        random.shuffle(self.documents)
    
    def flatten_docs(self):
        tokens = []
        for doc in self.documents:
            tokens.extend(doc)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
    
    def reset(self):
        self.shuffle_docs()
        self.flatten_docs()
        self.curr = 0

    def get_next_batch(self):
        
        buf = self.tokens[self.curr : self.curr + (self.B * self.T) + 1]

        x = buf[:-1].view(self.B,self.T)
        y = buf[1:].view(self.B,self.T)

        self.curr += self.B * self.T

        if (self.curr + (self.B * self.T) + 1) > len(self.tokens):
            self.reset()

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
train_loader = DataLoader(mini_batch_size, context_length, 'train')
val_loader = DataLoader(mini_batch_size, context_length, 'val')

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

# grokFast algo
def gradfilter_ema(
    m: torch.nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.9,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad}

    for n, p in m.named_parameters():
        if p.requires_grad:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads

# Finally, Train the model.
grads = None
val_loss = 0
train_losses = []
val_losses = []
for step in range(max_steps):
    t0 = time.time()

    # validation loss every 5 steps
    if step % 5 == 0:       
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for _ in range(min(grad_accum_steps,10)):
                x, y = val_loader.get_next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                    # compensate for accumulation due to 'mean' reduction in the loss function.
                    loss = loss / min(grad_accum_steps,10)
                    # compute gradients
                    val_loss+= loss.detach().cpu()
                            
        print(f'Validation Loss: {val_loss:.5f}')
    val_losses.append(val_loss)
    # lets also save the model every... 150 steps
    if step % 150 == 0:
        torch.save(model.state_dict(),f'./models/GPT-iter{step}.pt')

    # usual training loop
    model.train()

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
        loss_accum+= loss.detach().cpu()
        loss.backward()
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # accelerating slow gradients, for better generalization performance.
    # if step > 130:
    #     grads = gradfilter_ema(model, grads=grads)
    lr = get_lr(step)
    
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)
    tokens_processed_ps = train_loader.B * train_loader.T * grad_accum_steps / dt
    train_losses.append(loss_accum)
    print(f"step{step+1} | loss:{loss_accum.item():.5f} | lr: {lr:.3e} | t:{dt*1000:.2f}ms | norm: {norm.detach():.3f} | tok/s: {tokens_processed_ps:.2f}")

# save model
torch.save(model.state_dict(), 'GPT.pt')
# plot losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()
