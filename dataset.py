import numpy as np
import pandas as pd
import tiktoken 
import pickle

enc = tiktoken.get_encoding('gpt2')
eos = enc._special_tokens['<|endoftext|>']
dataset = pd.read_json('./dataset.jsonl', lines=True)
tokenized_dataset = []

token_count = 0 
for idx,p in dataset.iterrows():
    tokens = [eos]
    tk = enc.encode(p['text'])
    tokens.extend(tk)
    tokenized_dataset.append(tokens)
    # token_array = np.array(tokens).astype(np.uint16)
    # tokenized_dataset[token_count:token_count + len(tokens)] = token_array
    token_count+=len(tokens)

np.random.shuffle(tokenized_dataset)
split = 0.9
train_dataset = tokenized_dataset[:int(0.9 * len(tokenized_dataset))]
val_dataset = tokenized_dataset[len(train_dataset):]

with open('./dataset/dataset_train.pkl','wb') as f:
    pickle.dump(train_dataset,f)
with open('./dataset/dataset_val.pkl','wb') as f:
    pickle.dump(val_dataset,f)