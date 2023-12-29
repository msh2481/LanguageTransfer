import os
import random
from collections import defaultdict
from itertools import islice
from typing import Annotated, Any, Mapping, TypeVar, Literal

import circuitsvis
import einops as ein
import lightning.pytorch as pl  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from beartype.typing import Callable, Iterable
from beartype.vale import Is
from datasets import IterableDataset
from jaxtyping import Bool, Float, Int
from numpy import ndarray as ND
from torch import Tensor as TT
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from languages import dependencies_tokenizer


def seed_everything(seed):
    """
    Sets the seed for random, numpy and torch and torch.cuda.

    Parameters:
        seed (int): The seed value to set for the random number generators.

    Returns:
        None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.backends.cudnn.benchmark = False
    t.use_deterministic_algorithms(True)


def fetch_or_ask(var: str) -> str:
    """
    Fetches a variable from the environment or prompts the user for input and clears the output.

    Parameters:
        var (str): The name of the variable to fetch from the environment.

    Returns:
        str: The value of the variable.
    """
    from IPython.display import clear_output

    if var not in os.environ:
        val = input(f"{var}: ")
        clear_output()
        os.environ[var] = val
    return os.environ[var]


@typed
def show_string_with_weights(s: list[str], w: list[float] | Float[TT, "seq"]) -> None:
    """
    Displays a list of strings with each one colored according to its weight.

    Parameters:
        s (list[str]): The list of strings to display.
        w (list[float] | Float[TT, "seq"]): The list of weights for each token.

    Returns:
        None
    """
    from IPython.display import HTML, display
    from matplotlib import colormaps
    from matplotlib.colors import rgb2hex

    cmap = colormaps["coolwarm"]

    def brighten(rgb):
        return tuple([(x + 1) / 2 for x in rgb])

    if not isinstance(w, list):
        w = w.tolist()

    colors = [brighten(cmap(alpha)) for alpha in w]
    html_str_colormap = " ".join(
        [
            f'<span style="background-color: {rgb2hex(color)}; padding: 1px; margin: 0px; border-radius: 5px;">{word}</span>'
            for word, color in zip(s, colors)
        ]
    )
    display(HTML(html_str_colormap))


@typed
def module_device(model: nn.Module) -> str:
    return str(next(model.parameters()).device)


@typed
def tokenize(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[int] | Int[TT, "seq"],
    device: str = "cpu",
) -> dict[str, Int[TT, "batch seq"]]:
    if isinstance(prompt, str):
        result = tokenizer(prompt, return_tensors="pt")
    else:
        result = tokenizer(tokenizer.decode(prompt), return_tensors="pt")
    result["labels"] = result["input_ids"]
    assert (result["input_ids"] < len(tokenizer) - 2).all()
    return {name: value.to(device) for name, value in result.items()}


@typed
def generate_sample(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[int] | Int[TT, "seq"],
    max_new_tokens: int,
    keep_prompt: bool = False,
) -> str:
    inputs = tokenize(tokenizer, prompt, device=module_device(model))
    pad_token_id: int = tokenizer.pad_token_id or tokenizer.eos_token_id
    suffix: int = 0 if keep_prompt else len(inputs["input_ids"][0])
    output: Int[TT, "suffix"] = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=pad_token_id,
        bad_words_ids=[[pad_token_id]],
    )[0, suffix:]
    return tokenizer.decode(output.detach().cpu())


@typed
def get_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[int] | Int[TT, "seq"],
) -> Float[TT, "seq vocab"]:
    with t.no_grad():
        inputs = tokenize(tokenizer, prompt, device=module_device(model))
        logits: Float[TT, "seq vocab"] = model(**inputs).logits.squeeze(0)
        raw_lp: Float[TT, "seq"] = F.log_softmax(logits.cpu().detach(), dim=-1)
        return raw_lp.roll(1, dims=0)


@typed
def logprobs_to_losses(
    lp: Float[TT, "seq vocab"], labels: Int[TT, "seq"]
) -> Float[TT, "seq"]:
    return -lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


@typed
def get_loss(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str
) -> float:
    input_ids = tokenize(tokenizer, prompt, device=module_device(model))
    return model(**input_ids).loss.item()


@typed
def get_losses(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str
) -> Float[TT, "seq"]:
    """
    Remember, that the first element in the losses tensor is meaningless.
    """
    logprobs: Float[TT, "seq vocab"] = get_logprobs(model, tokenizer, prompt)
    ids: Int[TT, "seq"] = tokenize(tokenizer, prompt)["input_ids"][0]
    losses: Float[TT, "seq"] = logprobs_to_losses(logprobs, ids)
    return losses


@typed
def explore(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[int] | Int[TT, "seq"],
    n_tokens: int = 10,
    show_sample: bool = False,
) -> Float[TT, "seq"]:
    ids: Int[TT, "seq"] = tokenize(tokenizer, prompt)["input_ids"][0]
    logprobs: Float[TT, "seq vocab"] = get_logprobs(model, tokenizer, prompt)
    losses: Float[TT, "seq"] = logprobs_to_losses(logprobs, ids)

    # 0 for perfect prediction, 1 for infinite loss
    weights: Float[TT, "seq"] = (losses[-n_tokens:] / 2).tanh()
    tokens: list[str] = [tokenizer.decode(i) for i in ids[-n_tokens:]]
    show_string_with_weights(tokens, weights)

    if show_sample:
        sampled: str = generate_sample(model, tokenizer, ids[:-n_tokens], n_tokens)
        print(sampled)

    return losses


@typed
def explore_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: IterableDataset,
    n_samples: int = 32,
) -> None:
    losses = [
        explore(model, tokenizer, sample["input_ids"])[1:].mean().item()
        for sample in islice(dataset, n_samples)
    ]
    print(f"Mean loss: {sum(losses) / len(losses):.3f}")


@typed
def spectrum(data: Float[TT, "n d"]) -> Float[TT, "d"]:
    data = data - data.mean(0)
    vals = t.linalg.svdvals(data)
    return vals / vals.max()


@typed
def clusters(data: Float[TT, "n d"], max_clusters: int = 30) -> Float[TT, "k"]:
    from sklearn.cluster import KMeans

    one_cluster = (data - data.mean(0)).square().sum()
    results = [one_cluster, one_cluster]
    for k in range(2, max_clusters):
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100, algorithm="elkan")
        kmeans.fit(data)
        results.append(kmeans.inertia_)
    return t.tensor(results) / one_cluster


@typed
def singular_vectors(data: Float[TT, "n d"]) -> Float[TT, "n d"]:
    data = data - data.mean(0)
    return t.linalg.svd(data, full_matrices=False).Vh


@typed
def linear_regression_probe(
    embeddings: Float[TT, "n d"], features: Float[TT, "n"]
) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, features, test_size=0.2
    )
    model = Ridge().fit(X_train, y_train)
    y_pred = t.tensor(model.predict(X_test))
    return r2_score(y_test, y_pred)


@typed
def linear_classification_probe(
    embeddings: Float[TT, "n d"], features: Bool[TT, "n"]
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        features,
        test_size=0.2,
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)


@typed
def mixed_probe(
    embeddings: Float[TT, "n d"], features: pd.DataFrame
) -> dict[str, float]:
    result = {}
    for column in features.columns:
        dtype = str(features[column].dtype)
        y = t.tensor(features[column])
        if "float" in dtype:
            result[column] = linear_regression_probe(embeddings, y)
        else:
            result[column] = linear_classification_probe(embeddings, y)
    return result


@typed
def evaluate_cloze(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    cloze: str,
) -> float:
    assert prompt.count("#") == 1
    assert not prompt.startswith("#")
    prompt = prompt.replace("#", cloze)
    return get_loss(model, tokenizer, prompt)


@typed
def cloze_test(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tests: list[list[str]],
) -> Float[TT, "n"]:
    results = []
    for prompt, correct, incorrect in tests:
        loss_correct = evaluate_cloze(model, tokenizer, prompt, correct)
        loss_incorrect = evaluate_cloze(model, tokenizer, prompt, incorrect)
        results.append(loss_incorrect - loss_correct)
    return t.tensor(results)


@typed
def dependencies_del(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, seq: str
) -> Float[TT, "seq seq"]:
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


@typed
def sh(a: TT | tuple) -> list | tuple:
    if isinstance(a, tuple):
        return tuple(sh(x) for x in a)
    else:
        return list(a.shape)


@typed
def ls(a) -> str:
    if isinstance(a, TT) and a.shape == ():
        return ls(a.item())
    if isinstance(a, float):
        return f"{a:.2f}"
    if isinstance(a, int) or isinstance(a, bool):
        return str(int(a))
    if not hasattr(a, "__len__"):
        return str(a)
    brackets = "()" if isinstance(a, tuple) else "[]"
    children = [ls(x) for x in a]
    if any("(" in x or "[" in x for x in children):
        delim = "\n"
    else:
        delim = " "
    return delim.join([brackets[0]] + children + [brackets[1]])


@typed
def lincomb(
    alpha: float,
    x: TT | tuple,
    beta: float,
    y: TT | tuple,
) -> TT | tuple:
    if isinstance(x, tuple) and len(x) == 1:
        return lincomb(alpha, x[0], beta, y)
    if isinstance(y, tuple) and len(y) == 1:
        return lincomb(alpha, x, beta, y[0])
    if isinstance(x, TT) or isinstance(y, TT):
        assert x.shape == y.shape
        return alpha * x + beta * y
    else:
        assert len(x) == len(y)
        return tuple(lincomb(alpha, xi, beta, yi) for xi, yi in zip(x, y))


@typed
def activation_saver(
    inputs_dict: dict[str, TT | tuple],
    outputs_dict: dict[str, TT | tuple],
) -> Callable:
    @typed
    def hook(
        name: str, _module: nn.Module, input: TT | tuple, output: TT | tuple
    ) -> None:
        inputs_dict[name] = input
        outputs_dict[name] = output

    return hook


@typed
def chain_patcher(
    steps: list[str],
    real_inputs: dict[str, TT | tuple],
    inputs_dict: dict[str, TT | tuple],
    outputs_dict: dict[str, TT | tuple],
) -> Callable:
    @typed
    def hook(
        name: str, _module: nn.Module, input: TT | tuple, output: TT | tuple
    ) -> TT | tuple:
        inputs_dict[name] = input

        if (name in steps) and ("wte" not in name):
            name_for_delta = name
            if "attn" in name:
                name_for_delta = ".".join(name.split(".")[:-1] + ["ln_1"])
            if "mlp" in name:
                name_for_delta = ".".join(name.split(".")[:-1] + ["ln_2"])
            delta = lincomb(
                1.0, real_inputs[name_for_delta], -1.0, inputs_dict[name_for_delta]
            ).detach()
            if (
                len(output) > len(delta)
                and len(delta) == 1
                and isinstance(delta, tuple)
            ) or (isinstance(output, tuple) and isinstance(delta, TT)):
                a, b = output
                return (lincomb(1.0, a, 1.0, delta), b)
            output = lincomb(1.0, output, 1.0, delta)

        outputs_dict[name] = output
        return output

    return hook


class Hooks:
    @typed
    def __init__(
        self,
        module: nn.Module,
        hook: Callable[[str, nn.Module, TT, TT], TT],
    ) -> None:
        from functools import partial

        self.handles = []
        self.module = module
        for name, submodule in module.named_modules():
            if "." in name:
                self.handles.append(
                    submodule.register_forward_hook(partial(hook, name))
                )

    @typed
    def __enter__(self) -> None:
        pass

    @typed
    def __exit__(self, *_) -> None:
        for handle in self.handles:
            handle.remove()


@typed
def cross_loss(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str, label: str
) -> float:
    logprobs: Float[TT, "seq vocab"] = get_logprobs(model, tokenizer, prompt)
    ids: Int[TT, "seq"] = tokenize(tokenizer, label)["input_ids"][0]
    assert len(logprobs) == len(ids)
    losses: Float[TT, "seq"] = logprobs_to_losses(logprobs, ids)
    return losses[1:].mean().item()


@typed
def cross_loss_last(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str, label: str
) -> float:
    logprobs: Float[TT, "seq vocab"] = get_logprobs(model, tokenizer, prompt)
    ids: Int[TT, "seq"] = tokenize(tokenizer, label)["input_ids"][0]
    assert len(logprobs) == len(ids)
    losses: Float[TT, "seq"] = logprobs_to_losses(logprobs, ids)
    return losses[-1].item()


@typed
def get_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> tuple[dict[str, TT | tuple], dict[str, TT | tuple]]:
    input_dict: dict[str, TT | tuple] = {}
    output_dict: dict[str, TT | tuple] = {}
    with Hooks(model, activation_saver(input_dict, output_dict)):
        model(**tokenize(tokenizer, prompt))
    return input_dict, output_dict


@typed
def prediction_evolution(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> tuple[list[str], Float[TT, "n_layers seq vocab"]]:
    _, output_dict = get_activations(model, tokenizer, prompt)
    layer_names: list[str] = []
    layer_predictions: list[Float[TT, "seq vocab"]] = []

    @typed
    def process(name: str) -> None:
        activation = output_dict[name]
        if isinstance(activation, tuple):
            activation = activation[0]
        assert_type(activation, Float[TT, "batch seq d"])
        activation = activation.squeeze(0)
        logits = model.lm_head(model.transformer.ln_f(activation)).detach()
        assert_type(logits, Float[TT, "seq vocab"])
        logprobs = F.log_softmax(logits, dim=-1)

        layer_names.append(name[len("transformer.") :])
        layer_predictions.append(logprobs)

    process("transformer.wte")
    for layer in range(model.config.num_layers):
        process(f"transformer.h.{layer}.attn")
        process(f"transformer.h.{layer}.mlp")

    return layer_names, t.stack(layer_predictions)


@typed
def track_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    position: int,
) -> None:
    evolution = prediction_evolution(model, tokenizer, prompt)
    assert position != -1
    print("Target token:", tokenizer.tokenize(prompt)[position + 1])
    int_predictions = evolution[1][:, position, :].argmax(dim=-1).numpy()
    print("Predictions:", tokenizer.convert_ids_to_tokens(int_predictions))
    names = evolution[0]
    for token in range(len(tokenizer)):
        history = evolution[1][:, position, token].exp()
        if history[-1] < 0.01:
            continue
        str_token = tokenizer.convert_ids_to_tokens(token)
        plt.plot(names, history, label=f"{str_token} ({token})")
    history = evolution[1][:, position, ::2].exp().sum(dim=-1)
    plt.plot(names, history, label="open (::2)")
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


@typed
def get_layer_residuals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    layer_list: list[str],
) -> list[Float[TT, "seq d"]]:
    _, output_dict = get_activations(model, tokenizer, prompt)
    activations = []
    for i, layer_name in enumerate(layer_list):
        assert "lm_head" not in layer_name
        raw_output = output_dict[layer_name]
        is_tensor = isinstance(raw_output, TT)
        tensor = (raw_output if is_tensor else raw_output[0]).squeeze(0)
        if i > 1:
            activations.append(tensor - activations[-1])
        else:
            activations.append(tensor)
    return activations


@typed
def path_patching(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    real_prompt: str,
    corrupted_prompt: str,
    steps: list[str],
) -> float:
    real_inputs: dict[str, TT | tuple] = {}
    real_outputs: dict[str, TT | tuple] = {}
    with Hooks(model, activation_saver(real_inputs, real_outputs)):
        real_loss = cross_loss(model, tokenizer, prompt=real_prompt, label=real_prompt)

    corrupted_inputs: dict[str, TT | tuple] = {}
    corrupted_outputs: dict[str, TT | tuple] = {}
    with Hooks(
        model,
        chain_patcher(
            steps=steps,
            real_inputs=real_inputs,
            inputs_dict=corrupted_inputs,
            outputs_dict=corrupted_outputs,
        ),
    ):
        corrupted_loss = cross_loss(
            model, tokenizer, prompt=corrupted_prompt, label=real_prompt
        )
    return corrupted_loss - real_loss


@typed
def layer_importance_on_last_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    real_prompt: str,
    corrupted_prompt: str,
) -> dict[str, float]:
    assert len(tokenizer.tokenize(real_prompt)) == len(
        tokenizer.tokenize(corrupted_prompt)
    )

    real_inputs: dict[str, TT | tuple] = {}
    real_outputs: dict[str, TT | tuple] = {}
    with Hooks(model, activation_saver(real_inputs, real_outputs)):
        real_loss = cross_loss_last(
            model, tokenizer, prompt=real_prompt, label=real_prompt
        )

    results: dict[str, float] = {}

    @typed
    def process(name: str) -> None:
        corrupted_inputs: dict[str, TT | tuple] = {}
        corrupted_outputs: dict[str, TT | tuple] = {}
        with Hooks(
            model,
            chain_patcher(
                steps=[name],
                real_inputs=real_inputs,
                inputs_dict=corrupted_inputs,
                outputs_dict=corrupted_outputs,
            ),
        ):
            corrupted_loss = cross_loss_last(
                model, tokenizer, prompt=corrupted_prompt, label=real_prompt
            )
        results[name] = corrupted_loss - real_loss

    process("transformer.wte")
    for layer in range(model.config.num_layers):
        process(f"transformer.h.{layer}.attn")
        process(f"transformer.h.{layer}.mlp")

    return results


@typed
def patch_all_pairs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    real_prompt: str,
    corrupted_prompt: str,
    layer_list: list[str],
) -> Float[TT, "n_layers n_layers"]:
    n_layers = len(layer_list)
    losses: Float[TT, "n_layers n_layers"] = t.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            losses[i, j] = path_patching(
                model,
                tokenizer,
                real_prompt=real_prompt,
                corrupted_prompt=corrupted_prompt,
                steps=[layer_list[i], layer_list[j]],
            )
    return losses


@typed
def random_nested_prompt(n: int, n_types: int = 250) -> str:
    types = np.random.randint(0, n_types, (n,))
    ids = np.concatenate((2 * types, 2 * types[::-1] + 1))
    tokenizer = dependencies_tokenizer(vocab_size=2 * n_types)
    result = tokenizer.decode(ids)
    return result


@typed
def prompt_from_template(template: str, random: bool) -> str:
    stack = []
    result = []
    for c in template:
        if c == "(":
            bracket_type = np.random.randint(0, 250) if random else len(stack)
            stack.append(bracket_type)
            result.append(f"<{bracket_type + 1}")
        elif c == ")":
            bracket_type = stack.pop()
            result.append(f"{bracket_type + 1}>")
    return " ".join(result)


@typed
def describe_points(
    points: Float[TT, "n d"], svdvals=True, var=True, pca=False, cosine=True
) -> None:
    from sklearn.decomposition import PCA

    if svdvals:
        s = spectrum(points)
        print("Singular values:", s.sum().item(), s.numpy())
    if var:
        with_mean = ein.einsum(points, points, "n d, n d -> ").sqrt().item()
        white = points - points.mean(0)
        without_mean = ein.einsum(white, white, "n d, n d -> ").sqrt().item()
        print(
            "sqrt E[x x^T]:",
            with_mean,
            "\nsqrt E[(x-mu) (x-mu)^T]:",
            without_mean,
            "\nRatio:",
            without_mean / with_mean,
        )
    if pca:
        plt.figure(figsize=(5, 5))
        coords = PCA(n_components=2).fit_transform(points)
        plt.scatter(coords[:, 0], coords[:, 1])
        plt.show()
    if cosine:
        plt.figure(figsize=(5, 5))
        n = len(points)
        c = t.zeros(n, n)
        for i in range(n):
            for j in range(n):
                c[i, j] = t.cosine_similarity(points[i], points[j], dim=0)
        plt.imshow(c)
        plt.colorbar()
        plt.show()


@typed
def get_balances(prompt: str) -> Int[TT, "n_tokens"]:
    result = []
    for c in prompt:
        if c in "(<":
            result.append(1)
        elif c in ">)":
            result.append(-1)
    return t.tensor(result).cumsum(dim=0)


max_len = 256  # to avoid effects of local attention
bias = t.tril(t.ones((max_len, max_len), dtype=bool)).view(1, max_len, max_len)


@typed
def qk_to_attention_pattern(
    q: Float[TT, "n_q n_heads_d"], k: Float[TT, "n_k n_heads_d"], n_heads: int
) -> Float[TT, "n_heads n_q n_k"]:
    assert q.shape[0] == k.shape[0]
    seq_len = q.shape[0]
    assert seq_len <= max_len
    qs = ein.rearrange(q, "n_q (n_heads d) -> n_heads n_q d", n_heads=n_heads)
    ks = ein.rearrange(k, "n_k (n_heads d) -> n_heads n_k d", n_heads=n_heads)
    raw = ein.einsum(qs, ks, "n_heads n_q d, n_heads n_k d -> n_heads n_q n_k")
    causal_mask = bias[:, :seq_len, :seq_len]
    mask_value = t.finfo(raw.dtype).min
    raw = t.where(causal_mask, raw, mask_value)
    # seems GPTNeo implementation in transformers doesn't scale by sqrt(d)
    normalized = F.softmax(raw, dim=-1)
    return normalized


@typed
def show_patterns(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str, layer: int
) -> circuitsvis.attention.RenderedHTML:
    _, activations = get_activations(model, tokenizer, prompt)
    q = activations[f"transformer.h.{layer}.attn.attention.q_proj"].squeeze(0)
    k = activations[f"transformer.h.{layer}.attn.attention.k_proj"].squeeze(0)
    n_heads = model.config.num_heads
    pattern = qk_to_attention_pattern(q, k, n_heads=n_heads)

    # assert n_heads % 4 == 0
    # plt.figure(figsize=(8, 8))
    # for head in range(n_heads):
    #     plt.subplot(n_heads // 4, 4, head + 1)
    #     plt.gca().set_title(f"Head {head}")
    #     plt.imshow(pattern[head].detach())
    #     plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    tokens = tokenizer.tokenize(prompt)
    return circuitsvis.attention.attention_heads(pattern, tokens)


@typed
def input_output_mapping(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    input_layer: str,
    output_layer: str,
) -> tuple[Float[TT, "n_prompts seq_len d"], Float[TT, "n_prompts seq_len d"]]:
    @typed
    def squeeze_batch(
        x: TT | tuple,
    ) -> Float[TT, "total_tokens d"]:
        if isinstance(x, tuple):
            return squeeze_batch(x[0])
        assert x.shape[0] == 1
        return x.squeeze(0)

    input_list: list[Float[TT, "tokens d"]] = []
    output_list: list[Float[TT, "tokens d"]] = []
    for prompt in prompts:
        input_dict: dict[str, TT | tuple] = {}
        output_dict: dict[str, TT | tuple] = {}
        with Hooks(model, activation_saver(input_dict, output_dict)):
            model(**tokenize(tokenizer, prompt))
        input_list.append(squeeze_batch(input_dict[input_layer]).detach())
        output_list.append(squeeze_batch(output_dict[output_layer]).detach())
    return t.stack(input_list), t.stack(output_list)


@typed
def fit_linear(
    input: Float[TT, "total_tokens d"],
    output: Float[TT, "total_tokens d"],
    reg: Literal["l1", "l2"] = "l2",
    alpha: float = 0.0,
) -> nn.Linear:
    from sklearn.linear_model import Lasso, Ridge

    if reg == "l1":
        model = Lasso(alpha=alpha)
    else:
        model = Ridge(alpha=alpha)
    model.fit(input, output)
    linear = nn.Linear(input.shape[1], output.shape[1])
    linear.weight.data = t.tensor(model.coef_, dtype=t.float32)
    linear.bias.data = t.tensor(model.intercept_, dtype=t.float32)
    return linear


from collections import deque


class Tracker:
    @typed
    def __init__(self, max_window: int = 10**3):
        self.history: deque[float] = deque(maxlen=max_window)

    @typed
    def add(self, x: float):
        self.history.append(x)

    @typed
    def mean(self) -> float:
        return sum(self.history) / len(self.history)


@typed
def fit_module(
    model: nn.Module,
    input: Float[TT, "total_tokens d"],
    output: Float[TT, "total_tokens d"],
    lr: float,
    l1: float = 0.0,
    l2: float = 0.0,
    batch_size: int = 512,
    epochs: int = 10,
) -> None:
    optim = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    dataloader = t.utils.data.DataLoader(
        t.utils.data.TensorDataset(input, output),
        batch_size=batch_size,
        shuffle=True,
    )
    pbar = tqdm(range(epochs))
    loss_tracker = Tracker()
    for _ in pbar:
        for x, y in dataloader:
            optim.zero_grad()
            p = model(x)
            loss = F.mse_loss(p, y)
            if l1:
                params = list(model.parameters())
                loss = loss + l1 * sum(p.abs().mean() for p in params) / len(params)
            loss.backward()
            optim.step()
            loss_tracker.add(loss.item())
            pbar.set_postfix_str(f" loss: {loss_tracker.mean():.2f}")


class SparseAutoEncoder(nn.Module):
    @typed
    def __init__(self, in_features: int, h_features: int):
        super().__init__()
        self.in_features = in_features
        self.h_features = h_features
        self.weight = nn.Parameter(t.empty((in_features, h_features)))
        self.bias_before = nn.Parameter(t.empty((h_features,)))
        self.bias_after = nn.Parameter(t.empty((in_features,)))
        bound = (in_features * h_features) ** -0.25
        t.nn.init.normal_(self.weight, -bound, bound)
        t.nn.init.normal_(self.bias_before, -bound, bound)
        t.nn.init.normal_(self.bias_after, -bound, bound)

    @typed
    def encode(self, x: Float[TT, "... in_features"]) -> Float[TT, "... h_features"]:
        return F.leaky_relu(x @ self.weight + self.bias_before, negative_slope=0.01)

    @typed
    def decode(self, x: Float[TT, "... h_features"]) -> Float[TT, "... in_features"]:
        return x @ self.weight.T + self.bias_after

    @typed
    def forward(
        self, x: Float[TT, "... in_features"]
    ) -> tuple[Float[TT, "... in_features"], Float[TT, "... h_features"]]:
        code = self.encode(x)
        decoded = self.decode(code)
        return decoded, code


@typed
def fit_sae(
    model: SparseAutoEncoder,
    data: Float[TT, "total_tokens input_dim"],
    lr: float,
    l1: float = 0.0,
    alpha: float = 0.0,
    batch_size: int = 512,
    epochs: int = 10,
) -> None:
    optim = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    dataloader = t.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
    )
    pbar = tqdm(range(epochs))
    loss_tracker = Tracker()
    nonzero_tracker = Tracker()
    for _ in pbar:
        for x in dataloader:
            optim.zero_grad()
            p, z = model(x)
            loss = F.mse_loss(p, x)

            loss_tracker.add(loss.item())
            nonzero_tracker.add((z.abs() > 0.01).sum(dim=-1).float().mean().item())

            eps = 1e-8
            rel = z.abs() / (z.abs().max(dim=-1, keepdim=True).values + eps)
            loss = loss + l1 * rel.mean()

            sim = ein.einsum(z, z, "batch i, batch j -> i j")
            assert_type(sim, Float[TT, "dict_dim dict_dim"])
            nonorth = (sim - t.eye(model.h_features)).abs().mean()
            loss = loss + alpha * nonorth

            loss.backward()
            optim.step()
            pbar.set_postfix_str(
                f" loss={loss_tracker.mean():.2f}, nonzero={nonzero_tracker.mean():.2f} / {model.h_features}, nonorth={nonorth.item():.2f}"
            )


@typed
def eval_module(
    model: nn.Module,
    input: Float[TT, "total_tokens d"],
    output: Float[TT, "total_tokens d"],
) -> float:
    with t.no_grad():
        p = model(input)
        return F.mse_loss(p, output).item()


class Residual(nn.Module):
    @typed
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    @typed
    def forward(self, x: TT, *args, **kwargs) -> TT:
        return self.fn(x, *args, **kwargs) + x


class Wrapper(nn.Module):
    @typed
    def __init__(
        self, fn: nn.Module, ignore_extras: bool = True, append: tuple | None = None
    ):
        super().__init__()
        self.fn = fn
        self.ignore_extras = ignore_extras
        self.append = append

    def forward(self, x, *args, **kwargs):
        if self.ignore_extras:
            result = self.fn(x)
        else:
            result = self.fn(x, *args, **kwargs)
        if self.append is not None:
            return (result,) + self.append
        else:
            return result


class PrefixMean(nn.Module):
    @typed
    def __init__(self):
        super().__init__()

    @typed
    def forward(self, x: Float[TT, "... n 256"]) -> Float[TT, "... n 256"]:
        sums = x.cumsum(dim=-2)
        lens = t.arange(1, x.shape[-2] + 1).reshape([1] * (x.ndim - 2) + [-1, 1])
        return sums / lens
