import os
import random
from collections import deque
from itertools import islice
from typing import Annotated

import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import torch as t
from beartype import beartype as typed
from beartype.typing import Callable
from beartype.vale import Is
from datasets import load_dataset  # type: ignore
from matplotlib import pyplot as plt
from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast  # type: ignore


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


Even = Annotated[int, Is[lambda x: x % 2 == 0]]


@typed
def wordlevel_tokenizer(vocab: list[str]) -> PreTrainedTokenizerFast:
    vocab += ["[UNK]", "[PAD]"]
    model = models.WordLevel(
        {word: i for i, word in enumerate(vocab)},
        unk_token="[UNK]",
    )
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )


@typed
def dependencies_tokenizer(vocab_size: Even) -> PreTrainedTokenizerFast:
    vocab = [f"<{i//2+1}" if i % 2 == 0 else f"{i//2+1}>" for i in range(vocab_size)]
    return wordlevel_tokenizer(vocab)


@typed
def nested_dependencies_sequence(
    seq_len: Even,
    vocab_size: Even,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
    """
    Returns a sequence of `seq_len` tokens structured as nesting brackets
    of `vocab_size` different types. Token `2 * x` is an open bracket of
    type `x` and `2 * x + 1` is the corresponding closing one.
    """
    p_open = 0.4
    open_types: deque[int] = deque()
    data = [0] * seq_len
    for i in range(seq_len):
        should_open = t.rand(size=()) < p_open
        must_open = not open_types
        must_close = len(open_types) == seq_len - i
        if (should_open or must_open) and not must_close:
            tp = int(t.randint(low=0, high=vocab_size // 2, size=()))
            data[i] = 2 * tp
            open_types.append(tp)
        else:
            tp = open_types.pop()
            data[i] = 2 * tp + 1
    return tokenizer.decode(data)


@typed
def flat_dependencies_sequence(
    seq_len: Even,
    vocab_size: Even,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
    """
    Returns a sequence of `seq_len` matched tokens
    of `vocab_size` different types. Token `2 * x` is an open bracket of
    type `x` and `2 * x + 1` is the corresponding closing one.
    """
    p_open = 0.4
    open_types: list[int] = []
    data = [0] * seq_len
    for i in range(seq_len):
        should_open = t.rand(size=()) < p_open
        must_open = not open_types
        must_close = len(open_types) == seq_len - i
        if (should_open or must_open) and not must_close:
            tp = int(t.randint(low=0, high=vocab_size // 2, size=()))
            data[i] = 2 * tp
            open_types.append(tp)
        else:
            pos = int(t.randint(low=0, high=len(open_types), size=()))
            tp = open_types.pop(pos)
            data[i] = 2 * tp + 1
    return tokenizer.decode(data)


@typed
def flat_shuffle_sequence(
    seq_len: Even,
    group_len: int,
    vocab_size: Even,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
    """
    Returns a sequence of `seq_len` matched tokens of `vocab_size` different types.
    Open brackets are shuffled ranges of consecutive integers within groups of `group_len` tokens.
    Token `2 * x` is an open bracket of type `x` and `2 * x + 1` is the corresponding closing one.
    """
    p_open = 0.4
    open_types: list[int] = []
    data = [0] * seq_len
    shuffled_range: list[int] = []
    for i in range(seq_len):
        if not shuffled_range:
            range_start = int(t.randint(0, vocab_size // 2 - group_len, size=()))
            range_tensor = t.arange(range_start, range_start + group_len)
            shuffled_range = range_tensor[t.randperm(group_len)].tolist()
        should_open = t.rand(size=()) < p_open
        must_open = not open_types
        must_close = len(open_types) == seq_len - i
        if (should_open or must_open) and not must_close:
            tp = shuffled_range.pop()
            data[i] = 2 * tp
            open_types.append(tp)
        else:
            pos = int(t.randint(low=0, high=len(open_types), size=()))
            tp = open_types.pop(pos)
            data[i] = 2 * tp + 1
    return tokenizer.decode(data)


@typed
def generate_batch(
    generator: Callable[..., pa.RecordBatch],
    *args,
    **kwargs,
) -> Callable[[int, int], pa.RecordBatch]:
    def batch_generator(start: int, end: int) -> pa.RecordBatch:
        num_sequences = end - start
        return pa.RecordBatch.from_arrays(
            [pa.array(generator(*args, **kwargs) for i in range(num_sequences))],
            ["text"],
        )

    return batch_generator


@typed
def write_to_parquet(
    output_file: str,
    batch_size: int,
    total_size: int,
    generator: Callable[[int, int], pa.RecordBatch],
):
    schema = pa.schema([pa.field("text", pa.string())])
    with pq.ParquetWriter(output_file, schema) as writer:
        for start in tqdm(range(0, total_size, batch_size)):
            end = min(start + batch_size, total_size)
            batch = generator(start, end)
            writer.write_batch(batch)


def test_nested():
    seed_everything(1)
    dataset = load_dataset("Mlxa/nested", split="train", streaming=True)
    tok = dependencies_tokenizer(vocab_size=500)
    deltas = []
    for sample in islice(dataset, 1000):
        ids = tok(sample["text"])["input_ids"]
        back = tok.decode(ids)
        assert back == sample["text"]
        last_pos = {}
        for i, id in enumerate(ids):
            if id in last_pos:
                deltas.append(i - last_pos[id])
            last_pos[id] = i


# test_nested()

seq_len = 512
group_len = 8
vocab_size = 500
tokenizer = dependencies_tokenizer(vocab_size=vocab_size)
generator = flat_shuffle_sequence
batch_generator = generate_batch(generator, seq_len, group_len, vocab_size, tokenizer)

write_to_parquet(
    output_file="train.parquet",
    batch_size=10**3,
    total_size=2 * 10**6,
    generator=batch_generator,
)
write_to_parquet(
    output_file="test.parquet",
    batch_size=10**3,
    total_size=10**4,
    generator=batch_generator,
)
