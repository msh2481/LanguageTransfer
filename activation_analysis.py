from typing import Literal, NamedTuple

import circuitsvis
import einops as ein
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from jaxtyping import Bool, Float, Int
from torch import Tensor as TT
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GPTNeoForCausalLM

from hooks import Hooks, activation_modifier, activation_saver, get_activations
from language_modeling import get_logprobs, tokenize
from sparse_autoencoders import SparseAutoEncoder
from utils import Tracker


@typed
def qk_to_attention_pattern(
    q: Float[TT, "n_q n_heads_d"], k: Float[TT, "n_k n_heads_d"], n_heads: int
) -> Float[TT, "n_heads n_q n_k"]:
    max_len = 256  # to avoid effects of local attention
    bias = t.tril(t.ones((max_len, max_len), dtype=bool)).view(1, max_len, max_len)

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
def squeeze_batch(
    x: TT | tuple,
) -> TT:
    if isinstance(x, tuple):
        return squeeze_batch(x[0])
    assert_type(x, Float[TT, "1 ..."])
    return x.squeeze(0)


@typed
def input_output_mapping(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    input_layer: str,
    output_layer: str,
) -> tuple[Float[TT, "n_prompts seq_len d"], Float[TT, "n_prompts seq_len d"]]:
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


@typed
def compressed_activations(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    encoders: dict[str, SparseAutoEncoder],
) -> dict[str, Float[TT, "dict_dim"]]:
    _, activations = get_activations(model, tokenizer, prompt)
    return {
        name: encoder.encode(squeeze_batch(activations[name])).detach()
        for name, encoder in encoders.items()
    }


class FeatureEffect(NamedTuple):
    diff: dict[str, Float[TT, "seq d"]]
    base: dict[str, Float[TT, "seq d"]]


@typed
def feature_effect(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    position: int,
    directions: dict[str, tuple[SparseAutoEncoder, int]],
    eps: float = 1e-3,
) -> FeatureEffect:
    old_logprobs = get_logprobs(model, tokenizer, prompt).roll(-1, dims=0)
    assert_type(old_logprobs, Float[TT, "seq vocab"])
    _, old_activations = get_activations(model, tokenizer, prompt)
    old_activations = {k: squeeze_batch(v) for k, v in old_activations.items()}
    old_activations["lm_head"] = old_logprobs

    deltas = {}
    for name, (sae, direction) in directions.items():
        delta = t.zeros_like(old_activations[name])
        assert_type(delta, Float[TT, "seq d"])
        direction_vector = sae.decoder[direction]
        assert_type(direction_vector, Float[TT, "d"])
        delta[position] = direction_vector * eps
        deltas[name] = delta

    with Hooks(model, activation_modifier(deltas)):
        new_logprobs = get_logprobs(model, tokenizer, prompt).roll(-1, dims=0)
        assert_type(new_logprobs, Float[TT, "seq vocab"])
        _, new_activations = get_activations(model, tokenizer, prompt)
        new_activations = {k: squeeze_batch(v) for k, v in new_activations.items()}
        new_activations["lm_head"] = new_logprobs

    old_activations = {k: v for k, v in old_activations.items() if v.ndim == 2}
    difference = {
        name: new_activations[name] - old_activations[name] for name in old_activations
    }
    return FeatureEffect(diff=difference, base=old_activations)


@type
def feature_causes(
    model: GPTNeoForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    layer: str,
    sae: SparseAutoEncoder,
    direction: int,
) -> Float[TT, "seq"]:
    tokenized = tokenize(tokenizer, prompt)
    embeddings: Float[TT, "seq d"] = model.transformer.wte(
        tokenized["input_ids"]
    ).clone()
    del tokenized["input_ids"]

    input_dict: dict[str, TT | tuple] = {}
    output_dict: dict[str, TT | tuple] = {}
    embeddings.requires_grad_(True)
    with Hooks(model, activation_saver(input_dict, output_dict)):
        model(inputs_embeds=embeddings, **tokenized)
    full_activations = sae.encode(squeeze_batch(output_dict[layer]))
    assert_type(full_activations, Float[TT, "seq d"])
    feature_activation = full_activations[-1, direction]

    model.zero_grad()
    feature_activation.abs().backward()
    gradients = embeddings.grad.clone()
    assert_type(gradients, Float[TT, "seq d"])

    return gradients.norm(dim=1)


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
