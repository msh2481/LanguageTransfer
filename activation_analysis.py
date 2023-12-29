from typing import Literal

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
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from hooks import Hooks, activation_saver, get_activations
from language_modeling import tokenize


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
    data = data.clone()
    data /= data.std(dim=0, keepdim=True)
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
def compressed_activations(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    encoders: dict[str, SparseAutoEncoder],
):
    pass


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
