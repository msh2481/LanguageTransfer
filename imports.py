import os
import random
import shutil
import sys
import time
from collections import defaultdict, deque
from itertools import islice, product, cycle, chain
from typing import Annotated, Any

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
from jaxtyping import Float, Int
from lightning.pytorch.callbacks import LearningRateMonitor  # type: ignore
from numpy import ndarray as ND
from torch import Tensor as TT
from tqdm import tqdm


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
