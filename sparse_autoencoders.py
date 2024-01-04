from typing import Literal
import einops as ein
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from jaxtyping import Bool, Float, Int
from torch import Tensor as TT
from tqdm import tqdm
from utils import Tracker
from scipy.stats import special_ortho_group


@typed
def random_ortho(n: int, m: int) -> Float[TT, "n m"]:
    A = special_ortho_group.rvs(max(n, m))
    return t.tensor(A[:n, :m], dtype=t.float32)


@typed
def random_unit_columns(n: int, m: int) -> Float[TT, "n m"]:
    A = t.randn((n, m))
    return A / t.linalg.norm(A, axis=0, keepdims=True)


@typed
def toy_problem(
    n_samples: int,
    d_sparse: int,
    d_dense: int,
    ortho: bool,
    freq_decay: float,
    nonzero_rate: float,
) -> tuple[
    Float[TT, "n_samples d_dense"],
    Float[TT, "n_samples d_sparse"],
    Float[TT, "d_dense d_sparse"],
]:
    assert 0 < freq_decay <= 1
    assert 0 < nonzero_rate <= d_sparse

    directions = (
        random_ortho(d_dense, d_sparse)
        if ortho
        else random_unit_columns(d_dense, d_sparse)
    )
    frequencies = (freq_decay ** t.arange(d_sparse)).repeat(n_samples, 1)
    frequencies *= nonzero_rate / (frequencies.mean() * d_sparse)
    assert frequencies.max() < 1
    assert_type(frequencies, Float[TT, "n_samples d_sparse"])
    sparse_activations = t.bernoulli(frequencies) * t.rand((n_samples, d_sparse))
    dense_activations = ein.einsum(
        directions,
        sparse_activations,
        "d_dense d_sparse, n d_sparse -> n d_dense",
    )
    return dense_activations, sparse_activations, directions


@typed
def max_cosine_similarity(
    target: Float[TT, "n d_target"],
    learned: Float[TT, "n d_dict"],
) -> Float[TT, "d_target"]:
    assert not target.requires_grad
    assert not learned.requires_grad

    eps = t.tensor(1e-8)
    normed_targets = target / target.norm(dim=0, keepdim=True).max(eps)
    normed_dictionary = learned / learned.norm(dim=0, keepdim=True).max(eps)
    similarities = ein.einsum(
        normed_targets,
        normed_dictionary,
        "n d_target, n d_dict -> d_target d_dict",
    )
    return similarities.max(dim=1).values


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
        return F.relu(x @ self.weight + self.bias_before)

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
    data /= data.std()
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
