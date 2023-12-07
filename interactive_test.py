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
    tokenize,
    spectrum,
    clusters
)
from matplotlib import pyplot as plt

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

# %%
n = 1000
d = 5
c = 2
m = 3

centers = t.randn((c, d))
data = centers[t.randint(c, (n,))] + 0.01 * t.randn((n, m)) @ t.randn((m, d))

# %%
from utils import spectrum
s = spectrum(data)
print(s.sum())
plt.plot(s)
plt.show()

# %%
from utils import singular_vectors
vecs = singular_vectors(data)
print(vecs.shape)
for vec in vecs:
    print(vec.tolist())
    proj = data @ vec
    plt.plot(proj.numpy(), t.randn_like(proj).numpy(), "x")
    plt.show()
    plt.clf()

# %%
print(((centers[1] - centers[0]) / vecs[0]).tolist())
# %%
from utils import clusters
cl = clusters(data)
plt.plot(cl, "x", ms=5)
plt.ylim(0, 16e3)
plt.show()

# %%
from utils import mixed_probe
import pandas as pd

n, d = 1000, 5
emb = t.randn((n, d))
data = pd.DataFrame({
    "a": emb[:, 0] + 2 * emb[:, 1] + t.randn(n),
    "b": 0.1 * emb[:, 2] + 0.2 * emb[:, 3] + t.randn(n),
    "c": t.randn(n),
    "d": emb[:, 0] > emb[:, 1],
})
print(mixed_probe(emb, data))