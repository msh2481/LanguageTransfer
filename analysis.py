# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from beartype import beartype as typed
from torch import Tensor as TT
from jaxtyping import Float, Int
from typing import Mapping
from itertools import islice
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from languages import dependencies_tokenizer

%load_ext autoreload
%autoreload 2


# %%
model_names = [
    "Mlxa/tuned-nested-english",
    "Mlxa/tuned-flat-english",
    "Mlxa/tuned-flat_shuffle-english",
    "roneneldan/TinyStories-8M",
]

dataset = load_dataset("roneneldan/TinyStories", streaming=True)
# tokenizer = dependencies_tokenizer(vocab_size=500)
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
tokenizer.pad_token = tokenizer.eos_token
models = {
    name.split("/")[-1]: AutoModelForCausalLM.from_pretrained(name)
    for name in model_names
}


# %%
@typed
def tokenize_function(example: Mapping[str, str | int]) -> Mapping[str, list[int]]:
    result = tokenizer(
        example["text"], max_length=128, padding="max_length", truncation=True
    )
    result["labels"] = result["input_ids"]
    return result


test_size = 1000
tokenized_test = (
    dataset["validation"]
    .map(tokenize_function, batched=True)
    .remove_columns(["text"])
    .take(test_size)
)


# %%
@typed
def show_string_with_weights(
    s: list[str],
    w: list[float] | Float[TT, "seq"],
    width: int = 5,
) -> None:
    from IPython.display import HTML, display
    from matplotlib.colors import rgb2hex
    from matplotlib import colormaps

    cmap = colormaps["RdBu"]

    def brighten(rgb):
        return tuple([(x + 1) / 2 for x in rgb])

    if isinstance(w, TT):
        w = w.tolist()
    colors = [brighten(cmap(alpha)) for alpha in w]
    padded = [word[:width] + "&nbsp;" * max(0, width - len(word)) for word in s]
    html_str_colormap = " ".join(
        [
            f'<span style="background-color: {rgb2hex(color)}; padding: 0px; margin: 0px; border-radius: 5px;">{word}</span>'
            for word, color in zip(padded, colors)
        ]
    )
    display(HTML(f'<span style="font-family: comic mono">{html_str_colormap}</span>'))


@typed
def gather_logprobs(
    lp: Float[TT, "seq vocab"], ids: Int[TT, "seq"]
) -> Float[TT, "seq"]:
    return lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)


@typed
def explore(sample: Mapping[str, list[int]], l: int, r: int) -> None:
    device = "cuda" if t.cuda.is_available() else "cpu"
    for model in models.values():
        model.to(device)
    k = len(models)

    with t.no_grad():
        inputs = {k: t.tensor([v], device=device) for k, v in sample.items()}
        ids: Int[TT, "rng"] = inputs["input_ids"][0, l:r].cpu()
        tokens = [tokenizer.decode(i) for i in ids]
        lp: dict[str, Float[TT, "rng vocab"]] = {}
        lp_for_id: dict[str, Float[TT, "rng"]] = {}
        edge: dict[
            str, Int[TT, "rng"]
        ] = {}  # ~ Sharpe ratio relative to the entire ensemble

        for name, model in models.items():
            output = model(**inputs)
            raw_lp = F.log_softmax(output.logits.cpu().detach(), dim=-1).cpu()
            lp[name] = raw_lp.squeeze(0).roll(1, dims=0)[l:r]
            lp_for_id[name] = gather_logprobs(lp[name], ids)

        mean_lp = sum(lp.values()) / k
        mean_lp_for_id = sum(lp_for_id.values()) / k
        mean_std = 0.2 * sum(x.std() for x in lp_for_id.values()) / k

        print("ground truth and ensemble logprobs")
        weights = ((mean_lp_for_id - mean_lp_for_id.mean()) / mean_std).sigmoid()
        show_string_with_weights(tokens, weights)

        for name, model in models.items():
            print(name, "deviations and logprobs")
            edge[name] = (lp_for_id[name] - mean_lp_for_id) / mean_std
            weights = edge[name].sigmoid()
            deviations = [tokenizer.decode(i) for i in (lp[name] - 0 * mean_lp).argmax(-1)]
            show_string_with_weights(deviations, weights)
        print()


# %%
sentence = 'Spot saw a car and said, "That is a nice car!"'
tokenized = tokenizer(sentence)
explore(tokenized, l=0, r=20)


# TODO: like attention maps, but actually replacing a certain token with noise several times and measuring impacts
# %%

embeddings = AutoModelForCausalLM.from_pretrained("Mlxa/embeddings-flat_shuffle-english").get_input_embeddings().weight.detach().numpy()
print(embeddings.shape)

# %%
tokenizer = AutoTokenizer.from_pretrained("Mlxa/embeddings-nested-english")
all_tokens = [tokenizer.decode([i]) for i in range(len(embeddings))]

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2).fit(embeddings[:-2])
embeddings_pca = pca.transform(embeddings)

# %%
plt.figure(figsize=(20, 20), dpi=200)
selected = np.random.choice(len(embeddings), 40, replace=False)
plt.plot(embeddings_pca[selected, 0], embeddings_pca[selected, 1], "x", ms=5, color="blue")
for i in selected:
    plt.text(x=embeddings_pca[i, 0], y=embeddings_pca[i, 1], s=all_tokens[i])
plt.show()

# %%
from sklearn.cluster import KMeans
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)

# %%
plt.figure(figsize=(20, 20), dpi=200)
for i in range(n_clusters):
    plt.plot(embeddings_pca[kmeans.labels_ == i, 0], embeddings_pca[kmeans.labels_ == i, 1], "x", ms=5, label=str(i))
plt.legend()
plt.show()

# %%
import random
for cluster in range(n_clusters):
    belonging_tokens = [all_tokens[i] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == cluster]
    random.shuffle(belonging_tokens)
    print(cluster, belonging_tokens[:16])
# %%

def get_number(t):
    if "<" in t:
        return int(t[1:])
    elif ">" in t:
        return int(t[:-1])
    else:
        return None

def find_direction(f_weight):
    s = 0
    for e, t in zip(embeddings, all_tokens):
        s += f_weight(t) * e
    return s / (np.linalg.norm(s) + 1e-8)

# closedness = find_direction(lambda t: (">" in t) - ("<" in t))
# highness = find_direction(lambda t: get_number(t) - 125 if get_number(t) is not None else 0)

projected = embeddings @ np.array([closedness, highness]).T
plt.figure(figsize=(20, 20), dpi=200)
open_brackets = [i for i, t in enumerate(all_tokens) if "<" in t]
close_brackets = [i for i, t in enumerate(all_tokens) if ">" in t]
special = [i for i, t in enumerate(all_tokens) if "<" not in t and ">" not in t]

plt.plot(projected[:, 0], projected[:, 1], "x", ms=5, color="blue", label="open")
# plt.plot(projected[open_brackets, 0], projected[open_brackets, 1], "x", ms=5, color="blue", label="open")
# plt.plot(projected[close_brackets, 0], projected[close_brackets, 1], "x", ms=5, color="green", label="close")
plt.plot(projected[special, 0], projected[special, 1], "x", ms=10, color="red", label="special")
plt.legend()
plt.show()

# %%
len(all_tokens)

# %%
for i in range(len(all_tokens)):
    if projected[i, 0] > 3:
        print(all_tokens[i], projected[i])
# %%
# model = AutoModelForCausalLM.from_pretrained("Mlxa/embeddings-flat_shuffle-english")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-8M")
pairs = [
    ("I like water", "I water like"),
    ("She saw me", "She me saw"),
    ("Spot saw a car", "Spot a car saw"),
]
def get_loss(sentence):
    tokenized = tokenizer(sentence, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"]
    return model(**tokenized).loss.item()

for x, y in pairs:
    print(x, ":", y)
    print(get_loss(x), get_loss(y))

# %%

@typed
def get_lp_for_id(seq: str) -> Float[TT, "seq"]:
    tokens = tokenizer(seq, return_tensors="pt")
    tokens["labels"] = tokens["input_ids"]
    output = model(**tokens)
    ids: Int[TT, "rng"] = tokens["input_ids"][0]
    raw_lp = F.log_softmax(output.logits.cpu().detach(), dim=-1).cpu()
    lp = raw_lp.squeeze(0).roll(1, dims=0)
    return gather_logprobs(lp, ids)

@typed
def dependencies_del(model, seq: str) -> Float[TT, "seq seq"]:
    tokens_list = tokenizer(seq)["input_ids"]
    results = t.zeros((len(tokens_list), len(tokens_list)))
    original = get_lp_for_id(seq)

    with t.no_grad():
        for i in range(len(tokens_list)):
            without = tokens_list[:i] + tokens_list[i + 1 :]
            lp_for_id = get_lp_for_id(tokenizer.decode(without))
            for j in range(len(tokens_list) - 1):
                results[i, j + (j >= i)] = original[j + (j >= i)] - lp_for_id[j]

    print(original.int()) 
    return t.triu(results)

# %%
model = AutoModelForCausalLM.from_pretrained("Mlxa/embeddings-nested-english")
# model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-8M")
seq = "One day, a little girl named Lily found a needle in her room."
deps = dependencies_del(model, seq)
deps = deps.clip(-1, 5)
plt.imshow(deps)
plt.colorbar()
plt.show()

# %%
tokens = tokenizer(seq)["input_ids"]
for i, tok in enumerate(tokens):
    print(i, tokenizer.decode([tok]))
"""
0 One
1  day
2 ,
3  a
4  little
5  girl
6  named
7  Lily
8  found
9  a
10  needle
11  in
12  her
13  room
14 .
"""