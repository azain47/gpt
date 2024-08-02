## GPT2/GPT From Scratch
- Maintains most of the original params from the paper. 

- Didn't use weight sharing scheme of LM Head and token embeddings. (hence the increased model size at 163M params)

- Uses Shakespeare Toy dataset.

- Gradient Accumulation is done to an approximate of 32k tokens instead of 0.5M as stated in the paper.

- Optimizations done:
    - Flash Attention instead of usual attn calculation (10X speedup)
    - autocasting and reduced matmul precision; my gpu (RTX3060) supports bfloat16
    - changing params to a multiple of 2, like vocab size, batch size

> Also has code for Bigram Language Model, and attention variant of it as well.(gpt)