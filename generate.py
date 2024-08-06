from gpt2 import GPT2, GPTConfig
import torch
import torch.nn.functional as F
import tiktoken

config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embed=768
)

state_dict = torch.load('GPT.pt')
model = GPT2(config).to('cuda')
model.load_state_dict(state_dict)

model.eval()
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0)
x = tokens.to('cuda')

max_length = 250
temperature = 1
p = 0.95

# Top-P generation
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        # choose last token
        logits = logits[:,-1,:]

        probs = F.softmax(logits / temperature, dim=-1)
        
        # sort the probs, and calculate cumulative sum.
        probs_sort, probs_idx = torch.sort(probs,dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # the mask tells us the probabilities 
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0
        probs_sort.div_(probs_sort.sum(dim = -1, keepdim = True))
        ix = torch.multinomial(probs_sort, 1)

        xcol = torch.gather(probs_idx, -1, ix)

        x = torch.cat((x,xcol), dim=1)

decoded = enc.decode(x[0].tolist())
print(f'Top-P Generation:\n{decoded}')

x = tokens.to('cuda')

# Top-K generation
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)

        logits = logits[:,-1,:]

        probs = F.softmax(logits / temperature, dim=-1)
        
        topk_probs, topk_indices = torch.topk(probs,50)

        ix = torch.multinomial(topk_probs, 1)

        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x,xcol), dim=1)

decoded = enc.decode(x[0].tolist())
print(f'\n\nTop-K Generation:\n{decoded}')
