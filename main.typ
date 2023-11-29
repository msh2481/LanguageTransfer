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
- Flat shuffle: `<0 <3 3> <1 <2 1> 2> 0> <8 <7 7> <9 9> <6 6> 8> ...`
- Grammar induction: (have both recursive and context-free components): `A B A C A B A # a x b b a x c a x ...`

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
7. For each L2 language visualize outputs of different L1 models and levels of fine-tuning, including the model pre-trained on this L2 from scratch, if there is one. First, a list of sample generations. Second, visualize losses on samples from the dataset, especially those where the logits differ a lot. Third, compute distances between them based on divergences of their output distributions, show it with PCA.
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


== `nested`
This dataset consists of matched token pairs nested in each other. It is generated by the following algorithm:
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
```

Regarding the choice of parameters I follow @papadimitriou2023injecting and set `seq_len=512, vocab_size=500`, and generate approximately $10^9$ tokens in total, i.e. 2M sequences. I use uniform probabilities for all tokens.

An example word from it, with the matching pairs highlighted:

#let s = "<237 237> <249 249> <24 24> <162 162> <175 <211 <59 59> 211> 175> <233 233> <6 6> <13 13> <151 <106 <243 243> <83 83> 106> 151> <137 <207 <171 171> <50 50> 207> <107 107> <54 54> 137> <89 89> <71 71> <91 91> <75 75> <12 12> <127 <21 21> 127> <240 240> <229 <87 <212 <92 <198 <121 121> <101 101> 198> <33 33> <197 197> <184 184> 92> 212> 87> 229> <9 9> <12 12> <110 110> <112 112> <154 154> <50 50> <175 <221 221> 175> <220 220> <148 148> <141 141> <52 <115 115> <121 121> 52> <190 <196 196> 190> <229 <61 <93 <222 <63 63> 222> 93> <185 185> 61> <204 204> 229> <185 185> <151 151> <222 <245 245> 222> <156 156> <127 127> <34 34> <81 <47 <170 <178 178> 170> 47> 81> <14 14> <21 21> <246 246> <122 122> <228 228> <169 169> <118 <7 7> <131 131> <24 <164 <156 <29 29> 156> 164> 24> 118> <233 233> <65 65> <203 203> <169 169> <71 <72 72> <19 <22 22> 19> <227 227> <221 221> 71> <205 205> <10 <179 179> 10> <46 46> <249 249> <250 <115 <24 <165 165> <123 123> 24> 115> 250> <102 102> <76 <54 54> <206 <132 <235 235> <119 <221 221> 119> 132> 206> <248 <19 <210 210> 19> <184 <76 76> <59 59> 184> 248> <99 99> <191 <60 60> <86 <7 <197 197> <202 <240 <128 <170 170> 128> <63 63> 240> <18 18> 202> 7> <147 147> 86> 191> <176 <194 194> <88 88> 176> <77 77> <213 213> <216 216> 76> <130 <196 <244 244> <190 190> 196> 130> <44 <60 <122 <118 <59 59> 118> 122> <142 142> <57 57> <40 40> 60> 44> <168 <39 <76 76> <131 131> 39> 168> <86 <205 205> <65 65> 86> <16 <157 157> <103 <4 4> 103> <242 242> <96 <156 156> <24 24> 96> 16> <229 229> <137 <203 <7 <109 <84 <182 <170 170> <221 <211 211> <26 26> 221> <27 27> 182> 84> 109> 7> <84 84> <38 <88 <139 139> <183 183> 88> 38> 203> <107 <201 201> 107> 137> <72 72> <35 <178 178> <35 35> 35> <192 192> <125 125> <49 49> <151 151> <143 <44 44> 143> <3 3> <11 <4 <124 124> 4> <29 <197 <10 <190 190> <102 <14 14> <226 <79 <240 240> 79> 226> 102> 10> 197> <183 <180 180> 183> 29> 11> <46 46> <133 <203 203> 133> <175 175> <65 65> <120 120> <42 42> <115 <121 121> 115> <14 14> <8 8> <46 46> <56 <164 164> <35 35> <82 <155 <125 125> <20 <58 <122 <70 70> 122> 58> 20> 155> 82> 56> <151 151> <223 <123 123> 223> <44 <25 25> 44> <121 121> <115 115> <4 <11 <37 <212 212> <109 <49 49> 109> 37> <9 <30 <210 210> 30> 9> 11> <208 208> <135 <221 <200 200> 221> <101 101> <204 <8 <51 <177 177> 51> 8> 204> 135> 4>"

#let get_number(s) = {
    if "<" in s {
        return int(s.slice(1))
    } else {
        return int(s.slice(0, -1))
    }
}

#let get_color(s) = {
    let x = calc.rem(get_number(s), 360) * 5deg
    return color.hsl(x, 90%, 40%)

}

#for value in s.split() {
    text(raw(value + " "), fill: get_color(value), weight: 400, size: 6pt)
}

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

An example world:

#let s = "<216 216> <148 <46 148> 46> <65 <232 65> <225 <199 232> <123 225> <28 123> <106 106> <39 <88 <173 39> <64 199> 64> <132 173> 88> 132> <127 28> 127> <176 <95 <1 <110 95> 1> <143 176> <4 143> <127 <215 <157 <95 110> 127> 157> 95> 215> 4> <114 114> <167 167> <40 <134 <200 <40 40> <232 200> <131 <12 131> <35 <51 232> <242 <216 <207 242> <32 <20 40> 20> <96 32> 35> 51> 207> 12> 216> 96> 134> <31 <214 31> 214> <133 133> <76 76> <33 33> <62 62> <55 55> <182 <219 182> 219> <90 90> <122 122> <99 99> <243 243> <243 243> <108 <63 108> 63> <77 <68 77> 68> <195 195> <177 177> <148 148> <89 <193 193> 89> <34 34> <246 246> <151 151> <92 92> <166 166> <21 21> <194 194> <219 <105 219> <72 105> <150 <146 150> <123 146> 72> <162 <135 <125 135> <106 125> 123> 106> <229 <68 <47 229> 47> <174 174> 162> 68> <29 <238 238> <250 <136 250> <23 <152 152> 136> 23> 29> <189 <246 <54 <151 246> 189> <225 225> 151> 54> <197 197> <32 32> <50 50> <144 144> <139 <100 139> 100> <114 <38 38> 114> <60 <64 64> <102 <107 <155 155> 102> 107> 60> <243 <113 <55 113> 55> 243> <151 151> <59 <59 59> 59> <234 234> <152 <233 <197 <125 <9 <36 <9 <199 <129 152> <48 9> 125> <55 48> 129> 36> <65 65> 9> 199> 233> 197> 55> <8 <165 8> <94 94> 165> <147 <238 147> 238> <150 <107 107> <179 <116 <125 <6 150> <102 125> 179> 102> 6> <197 <195 <219 <146 197> 195> <208 <173 208> <47 146> 173> 219> <161 161> 116> <249 <151 <104 47> <70 104> <141 70> 141> <213 <136 <11 11> <234 <83 <78 83> 249> 151> 213> 136> <12 <117 <248 <240 234> 12> 78> <17 17> <62 62> <204 248> 117> 240> 204> <24 <14 24> <46 14> 46> <228 228> <26 26> <222 222> <52 52> <182 182> <131 <189 131> <72 189> 72> <111 111> <123 <236 123> 236> <12 12> <70 <124 70> 124> <146 <179 179> 146> <85 85> <218 <220 218> <193 220> 193> <187 187> <57 <134 57> <208 134> <235 208> <122 <177 177> <151 <20 151> 122> <103 20> <164 <54 54> 235> 103> 164> <224 224> <146 146> <144 144> <122 122> <95 <212 212> <204 95> 204> <20 <144 20> 144> <108 <116 116> 108> <35 35> <45 <5 45> 5> <248 <247 248> 247> <12 <245 12> <226 <94 226> 245> 94> <233 <244 233> 244> <231 231> <137 137> <140 <227 227> 140> <211 211> <90 90> <111 <165 111> <104 104> 165> <216 <137 137> 216> <112 <8 112> 8> <226 226> <157 157> <50 <178 <60 <57 178> <5 <205 50> 205> <174 60> <248 248> 174> 57> 5> <200 <218 200> 218> <87 87> <180 180> <169 169> <248 248> <132 <116 116> 132>"

#for value in s.split() {
    text(raw(value + " "), fill: get_color(value), weight: 400, size: 6pt)
}


== `flat_shuffle`

It is a combination of the previous dataset with the idea of implicit connections between tokens, as implemented in `Shuffle` languages from @chiang2022transferability. The idea behind it is that perhaps by adding more structure into the dataset, especially different kinds of structure, we can get a model with more interesting internal representations.

The only difference from the previous dataset is that types of opening brackets are now sampled not independently but in groups, where for each group tokens are first selected as a consecutive range of integers and then shuffled. Generating code:
```python
def flat_shuffle_sequence(
    seq_len: Even,
    group_len: int,
    vocab_size: Even,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
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
```

Example word:
#let s = "<58 <61 61> <54 58> 54> <57 57> <55 <56 <60 56> <59 55> 60> 59> <120 <121 120> 121> <125 <124 124> 125> <127 127> <123 123> <122 <126 126> 122> <147 147> <153 153> <148 148> <150 150> <151 151> <149 <152 152> 149> <154 <79 154> 79> <74 74> <75 75> <76 76> <81 <80 81> 80> <78 78> <77 77> <23 23> <21 21> <24 24> <26 26> <28 28> <25 <27 27> <22 22> <31 25> 31> <25 25> <26 <30 <29 <27 27> <32 32> <28 <127 30> 127> <130 130> 29> <134 134> <132 26> 132> 28> <131 <129 129> <128 <133 131> 133> <77 <75 77> 128> <78 <74 78> 75> 74> <80 <79 80> 79> <76 <81 <87 <82 <81 <80 80> 87> 76> 82> <84 81> 81> <85 84> <86 <83 86> 83> <32 <30 32> <34 30> 34> <29 85> <31 31> <33 29> <35 33> <36 <50 36> 35> <54 <51 <57 54> 51> 50> 57> <53 53> <55 55> <56 <52 56> 52> <166 166> <171 171> <170 170> <165 <169 <167 169> <168 167> 168> 165> <164 164> <194 194> <195 <190 <189 189> <196 <192 <193 195> 193> 190> 192> <191 196> 191> <110 110> <104 <106 106> <103 <107 <105 103> 104> <109 107> 105> <108 <128 109> 128> 108> <131 <127 131> 127> <132 132> <126 126> <125 125> <130 130> <129 <231 <236 129> <229 <232 231> <230 <233 230> <235 236> 229> 233> 235> 232> <234 <222 222> <218 218> 234> <223 223> <216 216> <217 217> <221 221> <220 220> <219 219> <6 6> <5 5> <8 <10 8> <12 <9 10> <11 11> 9> 12> <7 <110 7> 110> <115 115> <112 <114 112> 114> <111 111> <116 116> <109 109> <113 113> <101 101> <99 99> <102 102> <100 100> <105 105> <104 <98 <103 <42 <40 98> <39 <38 103> <41 <37 <43 104> <44 42> 44> 41> 40> 37> 38> 43> 39> <144 144> <150 150> <147 147> <146 146> <151 151> <149 149> <145 145> <148 <17 <21 17> 148> <15 21> <14 15> <20 <19 <18 20> <16 <33 14> <31 33> <37 <35 <36 19> 36> 37> <32 <34 <38 <159 <152 <158 <157 <155 <156 32> 35> <154 155> 159> <153 <98 157> <95 95> 31> 158> <99 <97 98> 154> <96 96> <100 <93 <94 94> <199 100> 18> 38> 199> 99> <204 97> 156> 204> <200 153> 93> 34> 152> <201 201> 200> 16> <198 198> <202 <203 202> 203> <197 197> <235 <229 <233 229> 233> 235> <232 232> <230 <234 <236 234> 230> <231 231> <91 91> 236> <95 <92 95> <94 <90 90> 92> <93 <96 96> 94> 93> <89 89> <186 186> <187 187> <185 185> <188 188> <192 192> <189 189> <191 191> <190 190> <115 115> <114 114> <112 112> <118 <111 111> 118> <116 116> <117 117> <113 113> <141 141> <142 142> <136 136> <140 140> <135 <138 135> <137 137> <139 139> 138>"

#for value in s.split() {
    text(raw(value + " "), fill: get_color(value), weight: 400, size: 6pt)
}

= Experiments

Pre-training:
- Finished early, when the training stagnated and loss came close to theoretical optimum.

Fine-tunings:
nested -> flat:
4.41 - 4.14 - 4.08 vs 3.78
flat -> nested:
3.47 - 3.34 - 3.34  vs 3.32
flat -> flat_shuffle:
2.46 - 2.36 - 2.15 vs 2.00
flat_shuffle -> flat:
3.82 - 3.80 - 3.76 vs 3.78