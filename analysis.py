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

embeddings = AutoModelForCausalLM.from_pretrained("Mlxa/brackets-flat_shuffle").get_input_embeddings().weight.detach().numpy()
print(embeddings.shape)

# %%
tokenizer = AutoTokenizer.from_pretrained("Mlxa/brackets-nested")
tokens = [tokenizer.decode([i]) for i in range(len(embeddings))]

# %%
# PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# pca = PCA(n_components=2).fit(embeddings[:-2])
# embeddings_pca = pca.transform(embeddings)

def get_number(t):
    if "<" in t:
        return int(t[1:])
    elif ">" in t:
        return int(t[:-1])
    else:
        return None

def find_direction(f_weight):
    s = 0
    for e, t in zip(embeddings, tokens):
        s += f_weight(t) * e
    return s / (np.linalg.norm(s) + 1e-8)

closedness = find_direction(lambda t: (">" in t) - ("<" in t))
highness = find_direction(lambda t: get_number(t) - 125 if get_number(t) is not None else 0)

projected = embeddings @ np.array([closedness, highness]).T
plt.figure(figsize=(20, 20), dpi=200)
open_brackets = [i for i, t in enumerate(tokens) if "<" in t]
close_brackets = [i for i, t in enumerate(tokens) if ">" in t]
special = [i for i, t in enumerate(tokens) if "<" not in t and ">" not in t]

plt.plot(projected[open_brackets, 0], projected[open_brackets, 1], "x", ms=5, color="blue", label="open")
plt.plot(projected[close_brackets, 0], projected[close_brackets, 1], "x", ms=5, color="green", label="close")
plt.plot(projected[special, 0], projected[special, 1], "x", ms=10, color="red", label="special")
plt.legend()
plt.show()

# %%
def synthethic_embedding(t: str) -> np.ndarray:
    x = 0.9 * ((">" in t) - ("<" in t))
    y = 0.006 * (get_number(t) - 125) + 0.2
    return x * closedness + y * highness

# %%
seq = ["<1", "<2", "2>", "<3", "<4"]
synth_tokens = [synthethic_embedding(t) for t in seq]
