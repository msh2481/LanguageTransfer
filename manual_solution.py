import torch as t
import matplotlib.pyplot as plt
import torch.nn.functional as F

t.manual_seed(0)
n_types = 24
dim = 16
factor = 2.0

type_embedding_matrix = t.randn((n_types, dim))
type_embedding_matrix /= type_embedding_matrix.norm(dim=1, keepdim=True)

n_pairs = 16
is_open = t.tensor([1] * n_pairs + [0] * n_pairs + [1] * n_pairs + [0] * n_pairs)
bracket_type = t.tensor(
    list(range(n_pairs))
    + list(range(n_pairs - 1, -1, -1))
    + list(range(n_pairs))
    + list(range(n_pairs - 1, -1, -1))
)

elevation = t.cumsum(2 * is_open - 1, dim=0) - is_open
weight = t.maximum(elevation, t.maximum(32 * (elevation - 5), 1024 * (elevation - 10)))
signed_weight = (2 * is_open - 1) * weight
type_embeddings = type_embedding_matrix[bracket_type]
weighted_embeddings = signed_weight.unsqueeze(-1) * type_embeddings
prefix_sums = t.cumsum(weighted_embeddings, dim=0)
denominator = t.pow(factor, -(elevation + is_open))
normalized = prefix_sums * denominator.unsqueeze(-1)
logits = normalized @ type_embedding_matrix.T
logits = F.softmax(logits * 100, dim=-1)
plt.imshow(logits)
# plt.savefig("img/mech.svg")
plt.show()
