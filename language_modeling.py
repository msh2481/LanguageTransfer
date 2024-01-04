from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from beartype.vale import Is
from datasets import IterableDataset
from jaxtyping import Bool, Float, Int
from torch import Tensor as TT
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from languages import dependencies_tokenizer


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
    from utils import show_string_with_weights

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
def get_balances(prompt: str) -> Int[TT, "n_tokens"]:
    result = []
    for c in prompt:
        if c in "(<":
            result.append(1)
        elif c in ">)":
            result.append(-1)
    return t.tensor(result).cumsum(dim=0)


@typed
def get_mean_balances(prompt: str) -> Float[TT, "n_tokens"]:
    result = []
    for c in prompt:
        if c in "(<":
            result.append(1)
        elif c in ">)":
            result.append(-1)
    sums = t.tensor(result).cumsum(dim=0)
    lens = t.arange(1, len(result) + 1)
    return sums / lens
