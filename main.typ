#import "template.typ": *
#show: ams-article.with(
  title: "Something about generalization",
  authors: (
    (
      name: "Mikhail Budnikov",
      uni: "Constructor University Bremen",
      email: "mbudnikov@constructor.university",
    ),
  ),
  abstract: [
    What knowledge is transfered between languages by Transformer models, for different types of transfer from zero-shot to complete fine-tuning?
  ],
  bibliography-file: "refs.bib",
)


= Plan


The framework is from @papadimitriou2020learning, using Transformers and fine-tuning their LayerNorm is from @lu2021pretrained, and testing tasks are from @kharitonov2020they. Interpretability as in @elhage2021mathematical.

*Overview*:
Take a single model, something like `GPT-2` or `TinyStories`. Pre-train it on different L1, then transfer to L2 and test there. Also, for synthetic languages, check what happens inside the pre-trained model, and how these mechanisms are used in transfer task.

*L1*:
- TinyStories
- Baseline: no pretraining, but mean and std matched to pretrained model
- Nested dependencies: `<1 <2 <3 3> 2> 1>`
- Flat dependencies: `<1 <2 <3 2> 3> 1>`
- Flat shuffle: `<0 <3 3> <1 <2 1> 2> 0> <8 <7 7> <9 9> <6 6> 8> ...`

*L2*:
- Everything from L1
- Count or memorization: `(aaa, bbb)` in train, `(aa, bb or bbb)` in test
- Memorize or add or multiply: `(aa, bbbb)` in train, `(a, bb or bbb or bbbb)` in test
- Hierarchical or linear: `(aabaa, b)` in train, `(aba, a or b)` in test
- Composition or memorization: `(a, a)`, `(b, b)`, `(thrice a, aaa)` in train, `(thrice b, bbb or b)` in test

*Transfer*:
- Inputs and outputs only
- Inputs, outputs and LayerNorm affine parameters
- Inputs, outputs, LayerNorm and the last Transformer block
- #strike[Parameter-efficient fine-tuning]
- #strike[Whole model]

= Experiments

1. Generate synthetic datasets for L1, $10^9$ tokens each.
  #note[Check the dependency length distribution.]
2. Implement tokenizers for each. (How it is done in other papers?)
  #note[Check that tokens are split as expected, and that the total number is right.]
2. Train `tiny-stories-8M` from `TransformerLens` on each, as in @papadimitriou2023injecting. Vocabulary size is different for different tasks. Should not take more than a GPU-day for each. Cloud computing to parallelize the process, DVC and containers to make it easy to deploy and cool to present.
  #note[Overfit one batch to check training code. Save checkpoints during training to study the emergence of structure later. Also save sample generations at each stage, to observe what the model learns on-the-fly.]
3. Generate synthethic datasets for L2. These will be quite small, maybe $10^3$ tokens. Tokenizers for these too.
4. Implement fine-tuning which saves in checkpoints only the trained parameters (if Lightning doesn't do it by default).
  #note[Overfit all types of fine-tuning on something small, check that heavier fine-tuning gives lower loss. ]
5. For each L1 model, and each L2 language, and each type of fine-tuning: Run the fine-tuning until convergence, save checkpoints on exponentially spaced timesteps. Each experiment should not take long, but now there are about $(binom(5, 2) + 5 times 5) times 3 = 105$ of them. Expect it to take approximately one more GPU-day. As only a small fraction of parameters is trained, checkpoints should not take much space.
6. Plot a $6 times 10 times 3$ matrix with transfer loss for each pair of languages and each level of fine-tuning.
7. For each L2 language visualize outputs of different L1 models and levels of fine-tuning. First, a list of sample generations. Second, compute distances between them based on divergences of their output distributions, show it with PCA.
8. Synthetic tasks are supposed to work better than random initialization, now it's time to find why. First, find positions which have small loss for transfer models and high loss for baseline. Pick representative examples from them.
9. Replace parts of the model with the untrained version with matched mean and std, check performance drops. This will highlight all necessary components.
10. Check attention patterns for the remaining heads, both on L1 and L2. Try to understand what they do and why is it important for the new task.

= Datasets
For each artificial language I generate words from it and save each as an indivudial sample. I chose Parquet format, because it is efficient in reading, and also allows to create dataset one batch per time and thus not be limited by the amount of RAM.

By iterative writing I mean the following approach:
```python
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
```


== `nested_small`
A small dummy dataset for quick tests. This one and the bigger version are generated by the following code:
== `nested`
This dataset consists of matched tokens nested in each other. It is generated by the following algorithm:
```python
def nested_dependencies_sequence(
    seq_len: Even,
    vocab_size: Even,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
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

def nested_dependencies_batch(
    seq_len: Even, vocab_size: Even
) -> Callable[[int, int], pa.RecordBatch]:
    def generator(start: int, end: int) -> pa.RecordBatch:
        num_sequences = end - start
        tokenizer = dependencies_tokenizer(vocab_size)
        return pa.RecordBatch.from_arrays(
            [
                pa.array(
                    nested_dependencies_sequence(
                        seq_len=seq_len,
                        vocab_size=vocab_size,
                        tokenizer=tokenizer,
                    )
                    for i in range(num_sequences)
                )
            ],
            ["text"],
        )

    return generator
```

Regarding the choice of parameters I follow @papadimitriou2023injecting and set `seq_len=512, vocab_size=500`, and generate approximately $10^9$ tokens in total. I use uniform probabilities for all tokens.

An example word from it:

#text(size: 8pt,

```
???
```
)

== `flat`
This dataset is almost like the previous one, but instead of stack-based grammar the decision which token to close now it made uniformly randomly among all currently open tokens. All parameters are the same.

```python
def flat_dependencies_sequence(
    seq_len: Even,
    vocab_size: Even,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
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
```

#note[Check that distributions match]

#text(size: 8pt,
```
???
```)