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
import nltk
from nltk.corpus import words

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')

word_set = set(words.words())
def get_word_properties(word):
    properties = {
        "start_space": word[0] == " ",
        "end_space": word[-1] == " ",
    }
    word = word.strip()
    if word in word_set:
        pos_tag = nltk.pos_tag([word])[0][1]
        properties["pos_tag"] = pos_tag
        if 'VB' in pos_tag:
            properties["tense"] = "present" if pos_tag in ["VBP", "VBZ"] else "past" if pos_tag == "VBD" else "infinitive"
        if 'NN' in pos_tag:
            properties["plural"] = pos_tag == "NNS"
            properties["proper"] = pos_tag == "NNP"

    return properties

word = "lied "
properties = get_word_properties(word)
print(properties)


# %%
get_model = lambda name: AutoModelForCausalLM.from_pretrained(name)
# from_scratch = get_model("roneneldan/TinyStories-8M")
# from_nested_emb = get_model("Mlxa/embeddings-nested-english")
# from_nested_tun = get_model("Mlxa/tuned-nested-english")
# from_flat_emb = get_model("Mlxa/embeddings-flat-english")
# from_flat_tun = get_model("Mlxa/tuned-flat-english")
# from_shuffle_emb = get_model("Mlxa/embeddings-flat_shuffle-english")
from_shuffle_tun = get_model("Mlxa/tuned-flat_shuffle-english")

# %%
prompt = 'Spot saw a car and said, "That is a nice car!"'
_ = explore(from_shuffle_tun, tokenizer, prompt, n_tokens=10**9, show_sample=False)

# %% [markdown]
# TODO:
# - test get_losses against .loss
# - finish property extraction
# - like attention maps, but actually replacing a certain token with noise several times and measuring impacts
# - list some properties (subject/object/verb, noun/adjective/verb, name, tense, starts from space, etc), then check how much variance each of them explains
# - indirect object identification, SVO order, gender synchronization, tense, using adjectives order, copying, copying in reverse, repeating, world model
# %%
get_embeddings = lambda model: model.get_input_embeddings().weight.detach()
embeddings = get_embeddings(from_shuffle_tun)
print(embeddings.shape)

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2).fit(embeddings[:-2])
embeddings_pca = pca.transform(embeddings)

# %%
plt.figure(figsize=(20, 20), dpi=200)
selected = np.random.choice(len(embeddings), 40, replace=False)
plt.plot(
    embeddings_pca[selected, 0], embeddings_pca[selected, 1], "x", ms=5, color="blue"
)
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
    plt.plot(
        embeddings_pca[kmeans.labels_ == i, 0],
        embeddings_pca[kmeans.labels_ == i, 1],
        "x",
        ms=5,
        label=str(i),
    )
plt.legend()
plt.show()

# %%
import random

for cluster in range(n_clusters):
    belonging_tokens = [
        all_tokens[i]
        for i in range(len(kmeans.labels_))
        if kmeans.labels_[i] == cluster
    ]
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
plt.plot(
    projected[special, 0],
    projected[special, 1],
    "x",
    ms=10,
    color="red",
    label="special",
)
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
