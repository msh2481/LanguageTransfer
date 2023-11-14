# %%
from imports import *
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
%load_ext autoreload
%autoreload 2


Even = Annotated[int, Is[lambda x: x % 2 == 0]]


@typed
def wordlevel_tokenizer(vocab: list[str]) -> Tokenizer:
    if "[UNK]" not in vocab:
        vocab = vocab + ["[UNK]"]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocab)}))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    return tokenizer


@typed
def dependencies_tokenizer(vocab_size: Even) -> Tokenizer:
    vocab = [f"<{i//2+1}" if i % 2 == 0 else f"{i//2+1}>" for i in range(vocab_size)]
    return wordlevel_tokenizer(vocab)


@typed
def nested_dependencies_sequence(
    seq_len: Even,
    vocab_size: Even,
    max_len: Even,
    tokenizer: Tokenizer,
) -> Int[TT, "seq_len"]:
    """
    Returns a sequence of `seq_len` tokens padded to `max_len` (plus BOS token) and structured
    as nesting brackets of `vocab_size` different types. Token `2 * x` is an
    open bracket of type `x` and `2 * x + 1` is the corresponding closing one.
    """
    p_open = 0.4
    open_types: deque[int] = deque()
    data = t.full(size=(max_len + 1,), fill_value=tokenizer.pad_token_id)
    data[0] = tokenizer.bos_token_id
    for i in range(1, seq_len - 1):
        should_open = t.rand(size=()) < p_open
        must_open = not open_types
        must_close = len(open_types) == seq_len - i - 1
        if should_open or must_open and not must_close:
            type = int(t.randint(low=0, high=vocab_size // 2, size=()))
            data[i] = 2 * type
            open_types.append(type)
        else:
            type = open_types.pop()
            data[i] = 2 * type + 1
    return data


@typed
def nested_dependencies_dataset(
    seq_number: int, max_len: Even, vocab_size: Even
) -> tuple[Int[TT, "seqences n"], Tokenizer]:
    tokenizer = dependencies_tokenizer(vocab_size)
    seq_len = t.randint(low=2, high=max_len // 2, size=(seq_number,)) * 2
    dataset = t.stack(
        [
            nested_dependencies_sequence(
                seq_len=int(seq_len[i]),
                vocab_size=vocab_size,
                max_len=max_len,
                tokenizer=tokenizer,
            )
            for i in range(seq_number)
        ]
    )
    return dataset, tokenizer


# @typed
# def write_to_parquet(output_file, batch_size, total_size):
#     schema = pa.schema([pa.field('number_str', pa.string())])
#     with pq.ParquetWriter(output_file, schema) as writer:
#         for start in range(0, total_size, batch_size):
#             end = min(start + batch_size, total_size)
#             batch = create_record_batch(start, end)
#             writer.write_batch(batch)

# def test_nested():
#     seed_everything(1)
#     vocab_size = 40
#     seq_len = 10
#     seq_number = 5
#     ids, tok = nested_dependencies_dataset(seq_number, seq_len, vocab_size)
#     print(ids)
#     print(*tok.decode(ids), sep="\n")
#     print(tok.encode(tok.decode(ids)))

# %%
# let's test dependency tokenizer
s = "<1 <2 2> 1>"
t = dependencies_tokenizer(10)
print(t.encode(s))
print(t.pad_token_id)
