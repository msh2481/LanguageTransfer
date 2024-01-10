import matplotlib.pyplot as plt
import torch as t
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from jaxtyping import Bool, Float, Int
from torch import Tensor as TT
import einops as ein


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
        plt.plot(coords[:, 0], coords[:, 1], "-o", lw=0.1, ms=1)
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
