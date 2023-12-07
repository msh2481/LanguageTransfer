from collections import Counter

import nltk
import pandas as pd
import torch as t
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("words")


dataset = load_dataset("roneneldan/TinyStories", streaming=True)
train_dataset = dataset["train"]

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
all_tokens = [tokenizer.decode([i]) for i in range(len(tokenizer))]

tokenized_dataset = train_dataset.map(lambda x: tokenizer(x["text"]), batched=True)
counter = Counter()
for tokens in tqdm(tokenized_dataset):
    counter.update(tokens["input_ids"])
d = [counter[i] for i in range(len(tokenizer))]
t.save(t.tensor(d), "word_freq.pt")

features = pd.DataFrame({"token": all_tokens, "frequency": d})
features["pos_tag"] = features["token"].apply(lambda x: nltk.pos_tag([x.strip()])[0][1])
features["start_space"] = features["token"].str.startswith(" ")
features.at[201, "token"] = "\\n"
features.to_csv("word_features.csv", index_label="id", escapechar="\\")
