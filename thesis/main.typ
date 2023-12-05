#import "template.typ": *
#show: ams-article.with(
  title: "Transfer of structural knowledge from synthethic languages",
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

#show outline: set heading(outlined: true)
#outline(indent: auto)
#pagebreak()

= Previous work

Machine learning systems are rapidly becoming more powerful. Data-driven methods can model increasingly more complex domains, so for example training a super-human image classifier @he2015delving or a language model which is super-human in terms of perplexity @shlegeris2022language is now mostly an engineering problem. However, even the amount of unlabeled data is limited and might be exhausted in this decade @villalobos2022will. This problem is even more pressing for the tasks which require human supervision or that are rarely found in web data. Which means that to get more powerful and universal machine learning systems simply collecting more diverse data is not enough and one need methods that can generalize efficiently under distribution shifts. Moreover, to be able to scale AI to superhuman levels safely without unexpected or undesired capabilities @hendrycks2023overview it is important to understand precisely in what ways these systems generalize.

In this work we take a bird's eye view on the question of why generalization occurs and how we can control it. Our main focus is on the natural language processing and especially on large language models, because they are currently the most universal machine learning systems, but many of the discussed ideas are relevant for machine learning in general.

To give a clear exposition of the topic we group all related ideas by the related stage of machine learning pipeline. In the end of each section we provide a summary of insights from the discussed works. The high-level list of considered questions is given below. 

== Data
Machine learning algorithms are designed to learn from data, but it is not always clear what exactly they learn from it. We consider such approaches as selecting high-quality data, using demonstrations of instruction following or more elaborate problem-solving behavior, generating data from another model or algorithm, using data from different task or modality and even using completely random data. Internal representations of pre-trained language models are also discussed. These observations highlight different ways in which the information used for training can affect the final model.

== Irrelevant data
One way to understand pre-training on language models is that we transfer some linguistic knowledge from a task wich lots of data available to a downstream task @han2021pre. As it turns out, there are many non-obvious effects of pre-training as well.

@papadimitriou2020learning show that pre-training LSTM @hochreiter1997long on structured but not linguistic data, such as MIDI music, Java code, or even nested parentheses, reduces its perplexity when testing on Spanish. It also works in the opposite direction, @lu2021pretrained get performance on different modalities comparable to training from scratch by fine-tuning only the input and output embeddings of the pre-trained GPT-2 @radford2019language. 

@sinha2021masked find that removing all information about word order from the pre-training phase does not affect the final performance very much, given a fine-tuning phase with the correct word order. 

@krishna2021does sample the training data from a completely artificial language, which consists of random n-grams, and observe that pre-training objectives that require processing this information somehow, such as copying sentences in the right order, still improve performance of the model on summarization tasks compared to randomly initialized version.

@maennel2020neural run training with entirely random labels and find that in some cases it still helps for further fine-tuning. Tracing the roots of this effect they find that the first layer of the network adapts to the data. More precisely, consider the weights from the inputs to a randomly selected neuron in the first hidden layer as a random variable, then during pre-training the eigenvectors of its covariance matrix align with the eigenvectors of covariance matrix of the data. @chowers2023cnns further show that the first layer converges to the whitening transformation for the training dataset.

There is evidence that pre-training leads optimization to a flat basin of the loss landscape. @mehta2021empirical confirm it and suggest as the reason why pre-trained models are more less susceptible to catastrophic forgetting during fine-tuning.  @neyshabur2020being also observe it and also show that models fine-tuned from the same checkpoint stay in the same basin.

== Internal representations

However, language models learn higher-level information as well. Recently, @gurnee2023language showed that pre-trained language models form quite detailed world models, in terms of both time and space representations. Namely, it is possible to extract information about where and when an event happened by linear projection from layer 50 activations on the corresponding token. @li2022emergent trained a language model to predict Othello moves and found that it represented the state of the board internally. @jin2023evidence demonstrate that language models trained on code have representations of program states and that quality of this representations is highly correlated with their ability to generate correct programs.

Another surprising finding is that language models tend to learn linear representations even when they are not explicitly directed to do so. A well-known example is Word2vec @mikolov2013efficient, which showed that applying vector arithmetic to word embeddings produces meaningful results. A more recent example in this direction, 
 @turner2023activation found that same trick works for GPT-2 activations. @nanda2023othello showed that Othello-GPT actually has linear world model, in @jin2023evidence the representations are linear as well.


== High-quality data
Current state-of-the-art language models are trained on vast amounts of text, which makes training them very expensive and soon can even become a bottleneck because the amount of data we have is finite. Regarding the topic of this study, an important question is what kind of data and how much of it is needed for the model to obtain certain capabilities, such as producing coherent English or zero-shot reasoning. In other words, how to construct the dataset to make the generalization easier.

@eldan2023tinystories show that by training on stories with very limited vocabulary it is possible to get a model with less than 10 million parameters which is still able to generate consistent and grammatically correct stories. 

@gunasekar2023textbooks extend this line of work towards programming domain and show that by carefully selecting data with the most educational value it is possible to significantly reduce the size of language models for code. @li2023textbooks check this for commonsense reasoning and also observe large gains. 

== Learning inductive bias from data

A useful inductive bias can be instilled into the model by pre-training on data that demonstrates it. @mccoy2020universal use pre-training on natural languages with certain properties by model-agnostic meta-learning @finn2017model to find which biases are needed to quickly acquire these languages. @wu2021lime design synthetic datasets requiring deduction, induction and abduction and pre-train on them to extract inductive bias for general mathematical reasoning. @lindemann2023injecting pre-train models to simulate finite state transducers given their description and achieve better generalization in NLP tasks with similar structure.

@mukherjee2023orca introduce a form of knowledge distillation suitable for language models, where a smaller model is trained on the explanations produced by a bigger one. Such rich training data helps to get better performance from smaller models, which implies boost in generalization. It can also be seen as transferring a superior inductive bias from the teacher model to the student.
#note[Add @mitra2023orca.]


#note[Use proper format for citet-style citations]
#note[Rewrite these parts]


== Properties of a trained model related to generalization

One setting for study of generalization is the i.i.d. case --- train and test samples are from the same distribution and independent, which allows to quantify generalization and provide lower bounds for it. Many properties are found to be be provide such lower bounds, though most of them are describing model complexity in some way:
-  Compressibility: @arora2018stronger, @lotfi2022pac
-  Weight norm: @bartlett1996valid, @wei2019data, @kawaguchi2017generalization
-  Flatness: @hochreiter1997flat, @bahri2021sharpness, @orvieto2022anticorrelated
-  Algorithmic stability: @chatterjee2022generalization, @bousquet2000algorithmic

Whenever a model is used on data on the real-world data, there almost always will be a distribution shift, training data can be completely representative of the actual distribution of inputs only in the simplest cases. So a more useful, though harder to measure, setting is out-of-distribution generalization. As formalized by @wolpert1996lack in no free lunch theorems, without additional assumptions about the problems, all learning algorithms are equally bad, and even such widespread technique as cross-validation will lose to anti-cross-validation in half of the cases. 

If one at least assumes that simple hypotheses should be preferred to complex ones, it is already enough to derive a general method for inductive inference, Solomonoff induction @solomonoff2009algorithmic. By averaging predictions of all possible models weighted by their Kolmogorov complexity it makes only a finite number of errors while predicting any computable sequence. @goldblum2023no points out that most real-world data sources indeed have such simplicity bias, or in other words, can be compressed, compared to the uniform distribution suggested by no free lunch theorems. An interesting observation is that modern deep learning models, including large language models, both trained and randomly initialized, also tend to show a preference for simple solutions @goldblum2023no, @valle2018deep.

== Avoiding too simple solutions <too_good>

One of the inductive biases is simplicity, also known as Occam's razor, meaning that for two features with equal predictive power the simpler one will be learned. Many works had shown that deep neural networks have it in some form, at least when trained with gradient descent @valle2018deep, @valle2018deep, @mingard2019neural. While in many cases it can be helpful to prevent overfitting, often there is also an obvious lower bound for the complexity of generalizable solution. In such cases it is desirable to exclude overly simple solutions, as they are likely to use spurious correlations from training data and fail on test data.

A possible way to deal with it is by training a smaller capacity model and the main model as an ensemble, so that the smaller model captures mostly the superficial patterns and the bigger learns the features that generalize better @clark2020learning. Also, as the superficial features tend to be learned first, one can train a biased model by giving increasingly more weight to the examples that it gets right, and train the main, unbiased model on the harder one @nam2020learning.

The notion of simplicity can be tailored for specific architectures or tasks. For language modeling an important property of a model is to capture long-distance dependencies. @malkin2021coherence increase the effective context length of a language model by subtracting output logits of short-context model from the main model outputs. @chuang2023dola focus on the differences between layers in a deep model. They note that outputs from the last layers tend to be more factually correct, and by selecting one of the earlier layers for contrasting, it is possible to improve the factuality even further. Namely, they select the layer which has the largest Jensen-Shannon divergence of next-word distributions with the final layer, and then subtract their logits.

== Controlling a known bias
Similar to methods for reducing simplicity bias, any known bias can be avoided by first training a separate model to capture features and patterns related to it, and then using its residuals to train the main model @he2019unlearn, or training the main model in an ensemble with the biased one @clark2019don.

Vision Transformers @dosovitskiy2020image achieved better data-efficiency  by training them to predict not only the correct answer, but also the answer given by a convolutional network @touvron2021training. Using two different teacher architectures brings even more benefits @ren2022co.

Of course, to be able to induce a certain inductive bias the model being pretrained has to be flexible enough to learn it, For example, LSTM gets less benefits from LIME pre-training @wu2021lime than Transformer @vaswani2017attention.

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
flat_shuffle -> english:
2.42 - 2.30 - 2.00 vs 1.19
english -> flat_shuffle:
2.77 - 2.62 - 2.11 vs 2.00
nested -> english:
2.82 - 2.65 - 2.37 vs 1.19
flat -> english:
2.74 - 2.56 - 2.35 vs 1.19
english -> flat:
4.28 - 4.16 - 3.76 vs 3.78

#note[12.9M out of 19.7M (i.e. 65%) parameters are embeddings.]