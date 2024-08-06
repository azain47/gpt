# Minimal GPT 
### <b>UPDATE-1, Aug 6 </b>
- Added Support for Rotary Positional Embeddings. Improved performance for poorly tokenized languages.
- Replaced LayerNorm with RMSNorm, however the difference seems minimal, on compute and performance, both.
- Added a Dataset preprocessor/generator. Tokenizes a dataset, shuffles and writes val and train datasets to pickle binaries.
- A better Dataloader, shuffles datasets, after each <b>epoch.*</b>
- Also added an Inferencing script, currently supports Top-P and Top-K generation strategies.
> epoch* : Epoch here means 1 full run through the dataset. If your dataset has 131072 tokens, and each training step you're going through 32768 tokens, then 32768 * 4 = 131072, 4 training steps will be equal to 1 epoch.
### Features
- Maintains most of the original params from the paper. 

- Didn't use weight sharing scheme of LM Head and token embeddings. (hence the increased model size at 163M params)

- Uses Shakespeare Toy dataset.

- Gradient Accumulation is done to an approximate of 32k tokens instead of 0.5M as stated in the paper.

- Optimizations done:
    - Flash Attention instead of usual attn calculation (10X speedup)
    - autocasting and reduced matmul precision; my gpu (RTX3060) supports bfloat16
    - changing params to a multiple of 2, like vocab size, batch size

> Also has code for Bigram Language Model, and attention variant of it as well.(gpt)
