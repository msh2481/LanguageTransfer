# %%
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from beartype import beartype as typed
from torch import Tensor as TT
from jaxtyping import Float, Int
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import explore


%load_ext autoreload
%autoreload 2

# %% [markdown]
# ## English (TinyStories)

# %%
dataset = load_dataset("roneneldan/TinyStories", streaming=True)
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
tokenizer.pad_token = tokenizer.eos_token
all_tokens = [tokenizer.decode([i]) for i in range(len(tokenizer))]

# %%
get_model = lambda name: AutoModelForCausalLM.from_pretrained(name)
big = get_model("roneneldan/TinyStories-33M")
from_scratch = get_model("roneneldan/TinyStories-8M")
from_nested_emb = get_model("Mlxa/embeddings-nested-english")
from_nested_tun = get_model("Mlxa/tuned-nested-english")
from_flat_emb = get_model("Mlxa/embeddings-flat-english")
from_flat_tun = get_model("Mlxa/tuned-flat-english")
from_shuffle_emb = get_model("Mlxa/embeddings-flat_shuffle-english")
from_shuffle_tun = get_model("Mlxa/tuned-flat_shuffle-english")

# %%
# ### Word-level information

# %%
import pandas as pd
features = pd.read_csv("word_features.csv", escapechar="\\")
features["frequency"] = features["frequency"].astype(float).apply(np.log1p)
features = pd.get_dummies(features, columns=["pos_tag"])
to_remove = [c for c in features.columns if (features[c] != False).sum() < 200]
features = features.drop(to_remove, axis=1)

# %%
get_embeddings = lambda model: model.get_input_embeddings().weight.detach()

emb_dict = {
    "scratch": get_embeddings(from_scratch),
    "nested": get_embeddings(from_nested_tun),
    "flat": get_embeddings(from_flat_emb),
    "shuffle": get_embeddings(from_shuffle_emb),
}

# %%
from utils import spectrum

plt.figure(figsize=(10, 10), dpi=200)
plt.title("Spectrum of covariation matrix")
for name, emb in emb_dict.items():
    sp = spectrum(emb)
    print(name, sp.sum().item())
    plt.plot(sp[:200], label=name)
plt.yscale("log")
plt.ylim(0.01, 1)
plt.legend()
plt.savefig(f"img/spectrum.svg")

# %%
# from utils import clusters

# plt.figure(figsize=(10, 10), dpi=200)
# plt.title("Inertia of KMeans")
# for name, emb in emb_dict.items():
#     print(name)
#     plt.plot(clusters(emb, max_clusters=100), label=name)
# plt.yscale("log")
# plt.ylim(0.1, 1)
# plt.legend()
# plt.savefig(f"img/clusters.svg")

# %%
from utils import singular_vectors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def analyze(name, n_vectors: int = 5, n_clusters: int = 10):
    emb = emb_dict[name]
    vecs = singular_vectors(emb)
    print("Singular vectors:", name)
    for i, v in enumerate(vecs[:n_vectors]):
        proj = emb @ v
        order = t.argsort(proj)

        print(f"#{i}, low:", [all_tokens[id] for id in order[:100:10]])
        print(f"#{i}, high:", [all_tokens[id] for id in order[-100::10]])
        plt.title("Distribution along this vector")
        plt.plot(proj.numpy(), t.randn_like(proj).numpy(), "x")
        plt.show()
        plt.clf()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(emb)
    labels = kmeans.labels_
    step = 100
    tsne = TSNE(n_components=2).fit(emb[::step]).embedding_
    for i in range(n_clusters):
        plt.plot(tsne[labels[::step] == i, 0], tsne[labels[::step] == i, 1], "x", label=f"#{i}")
    plt.legend()
    plt.show()
    plt.clf()
    for i in range(n_clusters):
        sample = [all_tokens[id] for id in t.randperm(len(all_tokens)) if labels[id] == i][:10]
        print(f"#{i}:", sample)

# %%
def analyze_probe(name):
    from utils import mixed_probe
    import json
    results = mixed_probe(emb_dict[name], features.drop(columns=["id", "token"]))
    print(json.dumps(results, indent=2))

# %%
analyze_probe("scratch")

# %%
analyze_probe("nested")

# %%
analyze_probe("flat")

# %%
analyze_probe("shuffle")

# %%
from utils import cloze_test
import json

with open("cloze_tasks.json") as f:
    tasks = json.load(f)

def analyze_cloze(model):
    results = {}
    for task_name, prompts in tasks.items():
        results[task_name] = cloze_test(model, tokenizer, prompts).mean().item()
    print(json.dumps(results, indent=2))

# %%
analyze_cloze(big)

# %%
print("Nested")
analyze_cloze(from_nested_tun)
analyze_cloze(from_nested_emb)

print("Flat")
analyze_cloze(from_flat_tun)
analyze_cloze(from_flat_emb)

print("Shuffle")
analyze_cloze(from_shuffle_tun)
analyze_cloze(from_shuffle_emb)

# %% [markdown]
# TODO:
# - indirect object identification, SVO order, gender synchronization, tense, using adjectives order, copying, copying in reverse, repeating, world model
# %%
from utils import get_logprobs

embeddings = get_embeddings(from_shuffle_tun)
print(embeddings.shape)

@typed
def dependencies_del(model, seq: str) -> Float[TT, "seq seq"]:
    tokens_list: list[int] = tokenizer(seq)["input_ids"]
    results = t.zeros((len(tokens_list), len(tokens_list)))
    original = get_logprobs(model, tokenizer, seq)

    with t.no_grad():
        for i in range(len(tokens_list)):
            without: list[int] = tokens_list[:i] + tokens_list[i + 1 :]
            lp_for_id = get_logprobs(model, tokenizer, without)
            for j in range(len(tokens_list) - 1):
                results[i, j + (j >= i)] = original[j + (j >= i)] - lp_for_id[j]

    return t.triu(results)

