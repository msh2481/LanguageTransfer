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
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from languages import dependencies_tokenizer
from utils import (
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
model_name = "roneneldan/TinyStories-8M"
dataset_name = "Mlxa/flat"
dataset = load_dataset(dataset_name, streaming=True)
tokenizer = (
    AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
    if dataset_name == "TinyStories"
    else dependencies_tokenizer(vocab_size=500)
)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# %%
model.resize_token_embeddings(len(tokenizer))
for name, param in model.named_parameters():
    param.requires_grad = "wte" in name or "wpe" in name

# %%
tokens_sample = tokenizer(next(iter(dataset["train"]))["text"])["input_ids"]
print(len(tokens_sample))
print(tokens_sample[:10])


# %%
if dataset_name == "TinyStories":

    @typed
    def tokenize_function(example: Mapping[str, str | int]) -> Mapping[str, list[int]]:
        result = tokenizer(
            example["text"], max_length=128, padding="max_length", truncation=True
        )
        result["labels"] = result["input_ids"]
        return result

else:

    @typed
    def tokenize_function(example: Mapping[str, str | int]) -> Mapping[str, list[int]]:
        result = tokenizer(example["text"])
        result["labels"] = result["input_ids"]
        return result


train_size = 100000
test_size = 1000
tokenized_train = (
    dataset["train"]
    .map(tokenize_function, batched=True)
    .remove_columns(["text"])
    .take(train_size)
)
tokenized_test = (
    dataset["validation" if dataset_name == "TinyStories" else "test"]
    .map(tokenize_function, batched=True)
    .remove_columns(["text"])
    .take(test_size)
)


# %%
@typed
def train(batch_size: int, lr: float) -> None:
    training_args = TrainingArguments(
        output_dir="trainer",
        fp16=False,
        per_device_train_batch_size=batch_size,
        torch_compile=False,
        learning_rate=lr,
        logging_steps=10,
        num_train_epochs=1,
        max_steps=train_size // batch_size,
        save_total_limit=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
    )
    trainer.add_callback(DVCLiveCallback())
    trainer.train()

# %%
explore_batch(model, tokenizer, tokenized_test)

# %%
# Fine-tuning only embeddings:
train(batch_size=8, lr=1e-2)

# %%
explore_batch(model, tokenizer, tokenized_test)

# %%
# Fine-tuning only embeddings and layernorms:
for name, param in model.named_parameters():
    if "ln" in name:
        print(f"{name} unfrozen")
        param.requires_grad = True
train(batch_size=8, lr=2e-3)

# %%
explore_batch(model, tokenizer, tokenized_test)

# %%
# Fine-tuning only embeddings, layernorms and the last block:
for name, param in model.named_parameters():
    if "h.7" in name:
        param.requires_grad = True
train(batch_size=8, lr=1e-3)

# %%
explore_batch(model, tokenizer, tokenized_test)

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
name = input("Model name, e.g. brackets-flat_shuffle: ")
model.push_to_hub(name)
tokenizer.push_to_hub(name)
# %%
1 / 0

# %%
import gc

gc.collect()
t.cuda.empty_cache()
