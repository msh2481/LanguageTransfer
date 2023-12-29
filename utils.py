import os
import random

import numpy as np
import torch as t
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from jaxtyping import Bool, Float, Int
from torch import Tensor as TT


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
