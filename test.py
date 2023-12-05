# %%
import unittest
from typing import TypeVar

import torch as t
from beartype import beartype as typed

from utils import (
    generate_sample,
    model_and_tokenizer,
    seed_everything,
    show_string_with_weights,
)


class UtilsTest(unittest.TestCase):
    @typed
    def setUp(self):
        seed_everything(0)
        if not (hasattr(self, "model") and hasattr(self, "tokenizer")):
            self.model, self.tokenizer = model_and_tokenizer(
                "Mlxa/brackets-flat_shuffle"
            )

    @typed
    def test_sampling(self):
        result = generate_sample(self.model, self.tokenizer, "<2 <8 <10", 10)
        self.assertEqual(result, "8> <4 4> 2> 8> 2> <9 <3 <7 <6")
        result = generate_sample(self.model, self.tokenizer, "<2 <8 <10", 10)
        self.assertEqual(result, "<9 <14 14> 10> 8> 2> <13 <12 <13 <9")


# %%
if __name__ == "__main__":
    seed_everything(0)
    show_string_with_weights(
        ["<2", "<8", "<10"],
        [0.0, 0.5, 1.0],
    )
