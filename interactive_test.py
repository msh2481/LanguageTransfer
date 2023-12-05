# %%
from typing import TypeVar, Mapping

import torch as t
from beartype import beartype as typed
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from itertools import islice

from utils import (
    generate_sample,
    seed_everything,
    show_string_with_weights,
    explore,
    get_logprobs,
    get_loss,
    explore_batch,
    tokenize
)

%load_ext autoreload
%autoreload 2

# %%
model_name = "Mlxa/brackets-nested"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
prompt = "<37 <135 <184 184> 135> 37>"
losses = explore(model, tokenizer, prompt, show_sample=False)
loss = get_loss(model, tokenizer, prompt)
print(losses.numpy())
assert losses[1:].mean().item() == loss

# %%
def tokenize_function(sample: Mapping[str, str | int]) -> Mapping[str, list[int]]:
    result = tokenizer(sample["text"], return_tensors="pt")
    result["labels"] = result["input_ids"]
    return result

dataset = load_dataset("Mlxa/nested", streaming=True)["train"].map(tokenize_function, batched=True).remove_columns(["text"])

# %%
explore_batch(model, tokenizer, dataset)

# %%
generate_sample(model, tokenizer, "<5", 10)

# %%
prompt = next(iter(dataset))["input_ids"]
input_ids = tokenize(tokenizer, prompt)
loss = model(
    **input_ids
).loss.item()
print(loss)