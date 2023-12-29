# %%
import os
from itertools import islice
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from beartype import beartype as typed
from datasets import load_dataset
from dvclive.huggingface import DVCLiveCallback
from IPython.display import clear_output
from jaxtyping import Float, Int
from torch import Tensor as TT
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from languages import dependencies_tokenizer
from language_modeling import (
    fetch_or_ask,
    explore_batch,
)

%load_ext autoreload
%autoreload 2

# %%
gdrive_token = fetch_or_ask("GDRIVE_CREDENTIALS_DATA")
os.environ[
    "DVC_STUDIO_TOKEN"
] = "isat_1mr9HNvqAB6xw8OJ3dXe5O9vMaKol59LCoA5gGP3eLY8NoSF8"

# %%
dataset = load_dataset("Mlxa/flat_shuffle")
model_name = "roneneldan/TinyStories-8M"
tokenizer = dependencies_tokenizer(vocab_size=500)
model = AutoModelForCausalLM.from_pretrained(model_name)

# %%
model.resize_token_embeddings(len(tokenizer))
for layer in model.parameters():
    layer.data = t.randn_like(layer.data) * 0.01

# %%
tokens_sample = tokenizer(dataset["train"][0]["text"])["input_ids"]
print(len(tokens_sample))
print(tokens_sample[:10])


# %%
@typed
def tokenize_function(example: Mapping[str, str | int]) -> Mapping[str, list[int]]:
    result = tokenizer(example["text"])
    result["labels"] = result["input_ids"]
    return result


subset_size = 60000
subset = dataset["train"].select(range(subset_size)).to_iterable_dataset()
tokenized = subset.map(tokenize_function, batched=True).remove_columns(["text"])

# %%
explore_batch(model, tokenizer, tokenized)

# %%
batch_size = 8

training_args = TrainingArguments(
    output_dir="trainer",
    per_device_train_batch_size=batch_size,
    learning_rate=1e-3,
    logging_steps=50,
    num_train_epochs=1,
    max_steps=subset_size // batch_size,
    save_total_limit=1,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)
trainer.add_callback(DVCLiveCallback())
trainer.train()

# %%
explore_batch(model, tokenizer, tokenized)

# %%
from huggingface_hub import notebook_login

notebook_login()

name = input("Model name, e.g. brackets-flat_shuffle: ")
model.push_to_hub(name)
tokenizer.push_to_hub(name)

# %%
1 / 0

# %%
import gc

gc.collect()
t.cuda.empty_cache()
