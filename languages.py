from imports import *


class SimpleTokenizer:
    """
    A tokenizer for a known vocabulary and with three special tokens.
    Based on https://github.com/callummcdougall/ARENA_2.0/blob/main/chapter1_transformers/exercises/part4_interp_on_algorithmic_model/brackets_datasets.py.
    """

    @typed
    def __init__(self, vocabulary: list[str]):
        self.start_token = "[start]"
        self.end_token = "[end]"
        self.pad_token = "[pad]"
        self.vocabulary = vocabulary + [
            self.start_token,
            self.end_token,
            self.pad_token,
        ]
        self.token_to_int = {c: i for i, c in enumerate(self.vocabulary)}
        self.int_to_token = {i: c for c, i in self.token_to_int.items()}
        self.start_token_id = self.token_to_int[self.start_token]
        self.end_token_id = self.token_to_int[self.end_token]
        self.pad_token_id = self.token_to_int[self.pad_token]

    @typed
    def encode(
        self, strs: str | list[str], max_len: int | None = None
    ) -> Int[TT, "batch max_len"]:
        if isinstance(strs, str):
            strs = [strs]

        tokens = [x.split() for x in strs]
        max_len = max_len or max(len(s) for s in tokens)
        ints = [
            [self.token_to_int[c] for c in s] + [self.pad_token_id] * (max_len - len(s))
            for s in tokens
        ]

        return t.tensor(ints, dtype=t.int)

    @typed
    def decode(
        self, tokens: Int[TT, "max_len"] | Int[TT, "batch max_len"]
    ) -> list[str]:
        if tokens.ndim == 1:
            tokens.unsqueeze_(0)
        return [
            " ".join(self.int_to_token[i.item()] for i in seq if i != self.pad_token_id)
            for seq in tokens
        ]


Even = Annotated[int, Is[lambda x: x % 2 == 0]]


@typed
def dependencies_tokenizer(vocab_size: Even) -> SimpleTokenizer:
    vocab = [f"<{i//2+1}" if i % 2 == 0 else f"{i//2+1}>" for i in range(vocab_size)]
    return SimpleTokenizer(vocab)


@typed
def nested_dependencies_sequence(
    seq_len: Even,
    vocab_size: Even,
    max_len: Even,
    tokenizer: SimpleTokenizer,
) -> Int[TT, "seq_len"]:
    """
    Returns a sequence of `seq_len` tokens padded to `max_len` and structured
    as nesting brackets of `vocab_size` different types. Token `2 * x` is an
    open bracket of type `x` and `2 * x + 1` is the corresponding closing one.
    """
    p_open = 0.4
    open_types: deque[int] = deque()
    data = t.full(size=(max_len,), fill_value=tokenizer.pad_token_id)
    data[0] = tokenizer.start_token_id
    data[seq_len - 1] = tokenizer.end_token_id
    for i in range(1, seq_len - 1):
        should_open = t.rand(size=()) < p_open
        must_open = not open_types
        must_close = len(open_types) == seq_len - i - 1
        # print(open_types)
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
) -> tuple[Int[TT, "seqences n"], SimpleTokenizer]:
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


def test_nested():
    seed_everything(1)
    vocab_size = 40
    seq_len = 10
    seq_number = 5
    ids, tok = nested_dependencies_dataset(seq_number, seq_len, vocab_size)
    print(ids)
    print(*tok.decode(ids), sep="\n")
    print(tok.encode(tok.decode(ids)))
