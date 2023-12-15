import os
import random
from itertools import islice
from typing import Annotated, Any, Mapping, TypeVar

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
        if name in steps:
            if "wte" in name:
                return output
            delta = lincomb(1.0, real_inputs[name], -1.0, input)
            if (
                len(output) > len(delta)
                and len(delta) == 1
                and isinstance(delta, tuple)
            ) or (isinstance(output, tuple) and isinstance(delta, TT)):
                a, b = output
                return (lincomb(1.0, a, 1.0, delta), b)
            output = lincomb(1.0, output, 1.0, delta)

        inputs_dict[name] = input
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
        tensor = (raw_output if isinstance(raw_output, TT) else raw_output[0]).squeeze(0)
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
def prompt_from_template(template: str) -> str:
    stack = []
    result = []
    for c in template:
        if c == "(":
            bracket_type = np.random.randint(0, 250)
            stack.append(bracket_type)
            result.append(f"<{bracket_type + 1}")
        elif c == ")":
            bracket_type = stack.pop()
            result.append(f"{bracket_type + 1}>")
    return " ".join(result)
