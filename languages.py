# %%
from imports import *
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import (
    models,
    pre_tokenizers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

%load_ext autoreload
%autoreload 2


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
    max_len: Even,
    tokenizer: PreTrainedTokenizerFast,
) -> Int[TT, "seq_len"]:
    """
    Returns a sequence of `seq_len` tokens padded to `max_len` (plus BOS token) and structured
    as nesting brackets of `vocab_size` different types. Token `2 * x` is an
    open bracket of type `x` and `2 * x + 1` is the corresponding closing one.
    """
    p_open = 0.4
    open_types: deque[int] = deque()
    data = t.full(size=(max_len,), fill_value=tokenizer.pad_token_id)
    for i in range(seq_len):
        should_open = t.rand(size=()) < p_open
        must_open = not open_types
        must_close = len(open_types) == seq_len - i
        if (should_open or must_open) and not must_close:
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
) -> Int[TT, "seqences n"]:
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
    return dataset


# @typed
# def write_to_parquet(output_file, batch_size, total_size):
#     schema = pa.schema([pa.field('number_str', pa.string())])
#     with pq.ParquetWriter(output_file, schema) as writer:
#         for start in range(0, total_size, batch_size):
#             end = min(start + batch_size, total_size)
#             batch = create_record_batch(start, end)
#             writer.write_batch(batch)

def test_nested():
    seed_everything(1)
    vocab_size = 40
    seq_len = 10
    seq_number = 5
    batch_ids = nested_dependencies_dataset(seq_number, seq_len, vocab_size)
    tok = dependencies_tokenizer(vocab_size)
    for ids in batch_ids:
        print(*tok.convert_ids_to_tokens(ids), sep="\t")
        assert (ids == tok(tok.decode(ids), return_tensors="pt")["input_ids"]).all()

# %%
test_nested()