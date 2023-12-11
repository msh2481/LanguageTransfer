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
    
Large Language Models are becoming increasingly capable and useful, but one of the bottlenecks is the amount of data required to train a model for a certain task, because the more data is used the more compute is need, and moreover, if the current scaling trends continue, soon there will not be enough high-quality data for even bigger models @villalobos2022will. At the same time, recent research shows @eldan2023tinystories @gunasekar2023textbooks that data efficiency can be vastly improved by using datasets tailored for better learning. Still, the high-level mechanisms of learning in LLMs remain poorly understood, as exemplified by suprising findings that pre-training a model on a simple algorithmic task can lead to improvements in language modelling @papadimitriou2020learning. Thus, better understanding of the mechanisms of transfer learning is an important open question.

In this work I focus on several algorithmic datasets and transfer learning from them to English, and employ diverse array of tehniques to analyze them deeper. In particular, I analyze the structure of the space of fine-tuned embeddings and the information contained in them and propose a plausible hypothesis regarding the algorithm implemented by the model internally. Also a new natural language understanding benchmark for tiny models is proposed and used to evaluate the capabilities of the fine-tuned models on a diverse set of tasks.
  ],
  bibliography-file: "refs.bib",
)

#show outline: set heading(outlined: true)
#outline(indent: auto)
#pagebreak()

= Introduction

The deep learning revolution in general and Transformer @vaswani2017attention architecture in partular led to the current situation where scaling language models is a reliable way of improving their performance. Although it is good to have scalable learning algorithms, this implies that the state-of-the-art models will always push the limits of available compute and data, which heavily concentrates the opportunities for meaningful researh in the hands of big tech companies with large compute clusters. Moreover, as a thourough analysis in @villalobos2022will shows, the amount of data, espeially high-quality text data, is limited and is going to become the main bottleneck in the following decades. 

Such circumstances motivate research into more data-effective learning algorithms and better understanding of the mechanisms of generalization and transfer learning. Humans are an obvious baseline here, because despite consuming orders of magnitude less data than the modern frontier models, they show non-trivial performance across many domains and even manage to outperform the machines in some of them despite all the recent algorithmic advances. Inspired by this, #cite(<huebner2021babyberta>, form: "prose") demonstrate that training RoBERTa @liu2019roberta on language acquisition data, together with some tweaks to model architecture and training, leads to 6000$times$ gains in data efficiency. In a similar vein, #cite(<eldan2023tinystories>, form: "prose") achieve significant model compression while retaining the ability to produce fluent and coherent English by using a generated dataset of children stories with smaller vocabulary. And #cite(<gunasekar2023textbooks>, form: "prose") find that filtering for data with higher educational value or creating such data is also very helpful.

So there is a growing body of evidence that the choice of data matters a lot and simply scraping the data from the web is suboptimal. However, there is a limited understanding of what properties of the data are important in different training stages. It is well illustrated by a series of findings challenging the common assumptions about the role of data in pre-training. 

#cite(<papadimitriou2020learning>, form: "prose") show that pre-training LSTM @hochreiter1997long on structured but not linguistic data, such as MIDI music, Java code, or even nested parentheses, reduces its perplexity when testing on Spanish. It also works in the opposite direction, #cite(<lu2021pretrained>, form: "prose") get performance on different modalities comparable to training from scratch by fine-tuning only the input and output embeddings of the pre-trained GPT-2 @radford2019language. #cite(<sinha2021masked>, form: "prose") find that removing all information about word order from the pre-training phase does not affect the final performance very much, given a fine-tuning phase with the correct word order. #cite(<krishna2021does>, form: "prose") sample the training data from a completely artificial language, which consists of random n-grams, and observe that pre-training objectives that require processing this information somehow, such as copying sentences in the right order, still improve performance of the model on summarization tasks compared to randomly initialized version. #cite(<maennel2020neural>, form: "prose") run training with entirely random labels and find that in some cases it still helps for further fine-tuning.

However, research in this direction is currently mostly concentrated on reporting surprising observations rather than providing explanations for them and building a general theory. For example, below is the main figure from @papadimitriou2020learning, and that paper is entirely devoted to discussing the differences in perplexity after pre-training on different languages.
#image("../img/tilt.png")

While making such observations is, without doublt, and important step, more data-efficiency advances can be expected if the cause of these observations is better understood. Therefore, this work attempts to make a small step in that direction and introduces more techniques that can be used to study the mechanisms of transfer learning.

First, as can be seen on the diagram above, different pre-training datasets, even if they are all not related to the target task, lead to different final performance. It suggests an idea that some languages are intrinsically more complex, or perhaps more similar to the target language. To better understand the structure of the language space I introduce a new synthethic language by combining ideas from the previous work and use it  as well as two already existing synthethic datasets to pre-train the models. Then I fine-tune them to English using three different levels of fine-tuning and observe how the performance depends on the language and the allowed flexibility of fine-tuning. From theoretical side, I provide an algorithm which might be used by the models trained on this datasets and discuss the implications for the difficulty of these languages and their effect on model parameters during pre-training.

Second, as one of the settings for transfer learning involves fine-tuning only embeddings, they are the natural target for investigation. I explore the structure of the learned embeddings, namely, the spectrum of their singular values which is a way to understand the effective dimensionality of the data, and the performance of KMeans clustering on them with different number of clusters which is a way to check how uniformly the embeddings are distributed. To check what information is contained in the embeddings I train linear probes to predict certain features of the words given their embeddings. Linear probes are a popular interpretability technique, but to the best of my knowledge they have not been used in this context, i.e. to study the embeddings of models pre-trained on different datasets and fine-tuned to the same task.

Finally, I evaluate the performance of these models in natural language understanding. As existing NLU datasets such as GLUE @wang2018glue and MMLU @hendrycks2020measuring are designed for more capable models, I use GPT-4 @openai2023gpt4 to generate a simplar benchmark consisting of 12 diverse subtasks. 


= Background and motivation

#note[Rewrite these parts]

Machine learning systems are rapidly becoming more powerful. Data-driven methods can model increasingly more complex domains, so for example training a super-human image classifier @he2015delving or a language model which is super-human in terms of perplexity @shlegeris2022language is now mostly an engineering problem. However, even the amount of unlabeled data is limited and might be exhausted in this decade @villalobos2022will. This problem is even more pressing for the tasks which require human supervision or that are rarely found in web data. Which means that to get more powerful and universal machine learning systems simply collecting more diverse data is not enough and one need methods that can generalize efficiently under distribution shifts. Moreover, to be able to scale AI to superhuman levels safely without unexpected or undesired capabilities @hendrycks2023overview it is important to understand precisely in what ways these systems generalize.

In this work we take a bird's eye view on the question of why generalization occurs and how we can control it. Our main focus is on the natural language processing and especially on large language models, because they are currently the most universal machine learning systems, but many of the discussed ideas are relevant for machine learning in general.

To give a clear exposition of the topic we group all related ideas by the related stage of machine learning pipeline. In the end of each section we provide a summary of insights from the discussed works. The high-level list of considered questions is given below. 

== Data
Machine learning algorithms are designed to learn from data, but it is not always clear what exactly they learn from it. We consider such approaches as selecting high-quality data, using demonstrations of instruction following or more elaborate problem-solving behavior, generating data from another model or algorithm, using data from different task or modality and even using completely random data. Internal representations of pre-trained language models are also discussed. These observations highlight different ways in which the information used for training can affect the final model.

== Irrelevant data
One way to understand pre-training on language models is that we transfer some linguistic knowledge from a task wich lots of data available to a downstream task @han2021pre. As it turns out, there are many non-obvious effects of pre-training as well.
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

== Open questions

As mentined above, there were many works that studied transfer learning from artificial datasets to natural language, for variety of model architectures, datasets and fine-tuning methods. However, they all report only superficial metrics, like losses and perplexities. To shed light on what actually happens during such transfer, in this work I use diverse analysis techniques to get richer observations of this process and as a result, hopefully, get deeper insights. In particular, I use the following methods:
- Compare the complexity of synthethic languages by training a model on one language and then fine-tuning to another.
- To get more feedback from these experiments, fine-tuning is done in several stages, each one allowing more complexity to be learned.
- Study the structure of the embedding space, in terms of the singular values of correlation matrix and by running KMeans algorithm on it with different number of clusters.
- Train linear probes to predict grammatical features of the words represented by tokens and their frequency, thus showing what information might be contained in the embeddings.
- Introduce a new natural languge understanding dataset tailored for less capable models which benchmarks the model performance on 12 diverse subtasks.

= Experiments

== Synthethic languages
Following hyperparameter choices from @papadimitriou2023injecting, for each of the languages descrbed below I use sequence length of $512$, vocabulary size of $500$, and generate $2 dot 10^6$ sequences so the total size of the dataset is approximately $10^9$ tokens in each case.



=== `nested`
The predcessor of this dataset was introduced in @papadimitriou2020learning. They used stack-based grammar to generate sequences, where each token occurs twice and two pairs of tokens either do not intersect or one is nested in another. In other words, balanced bracket sequence with multiple types of brackets.

@ri2022pretraining suggested to use different tokens for opening and closing brackets, and found improved performance. I chose to implement this version, so there are $250$ tokens for open brackets and $250$ tokens for closing ones.

Regarding the sampling algorithm for this language, tokens are generated sequentially and on each step a random decision is made whether to open a new bracket or to close an existing one. If the stack of open brackets is empty or there is not enough space before the end of sequence, there is only one option. In other cases an openning bracket is chosen with probability $0.4$ and then the type of bracket is sampled uniformly.


Generation algorithm in Python:
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
#pagebreak()
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

#box(stroke: 1pt + blue, radius: 5pt, inset: 10pt)[
#for value in s.split() {
    text(raw(value + " "), fill: get_color(value), weight: 400, size: 6pt)
}
]

=== `flat`
This language is similar to the previous one and was introduced and enhanced in the same works. The only difference is that the nesting property can be violated. 

In terms of sampling it means that when a bracket should be closed, now there is more than one possibility. I select the bracket to close uniformly from all currently open ones.

Generation algorithm in Python:
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


#let s = "<216 216> <148 <46 148> 46> <65 <232 65> <225 <199 232> <123 225> <28 123> <106 106> <39 <88 <173 39> <64 199> 64> <132 173> 88> 132> <127 28> 127> <176 <95 <1 <110 95> 1> <143 176> <4 143> <127 <215 <157 <95 110> 127> 157> 95> 215> 4> <114 114> <167 167> <40 <134 <200 <40 40> <232 200> <131 <12 131> <35 <51 232> <242 <216 <207 242> <32 <20 40> 20> <96 32> 35> 51> 207> 12> 216> 96> 134> <31 <214 31> 214> <133 133> <76 76> <33 33> <62 62> <55 55> <182 <219 182> 219> <90 90> <122 122> <99 99> <243 243> <243 243> <108 <63 108> 63> <77 <68 77> 68> <195 195> <177 177> <148 148> <89 <193 193> 89> <34 34> <246 246> <151 151> <92 92> <166 166> <21 21> <194 194> <219 <105 219> <72 105> <150 <146 150> <123 146> 72> <162 <135 <125 135> <106 125> 123> 106> <229 <68 <47 229> 47> <174 174> 162> 68> <29 <238 238> <250 <136 250> <23 <152 152> 136> 23> 29> <189 <246 <54 <151 246> 189> <225 225> 151> 54> <197 197> <32 32> <50 50> <144 144> <139 <100 139> 100> <114 <38 38> 114> <60 <64 64> <102 <107 <155 155> 102> 107> 60> <243 <113 <55 113> 55> 243> <151 151> <59 <59 59> 59> <234 234> <152 <233 <197 <125 <9 <36 <9 <199 <129 152> <48 9> 125> <55 48> 129> 36> <65 65> 9> 199> 233> 197> 55> <8 <165 8> <94 94> 165> <147 <238 147> 238> <150 <107 107> <179 <116 <125 <6 150> <102 125> 179> 102> 6> <197 <195 <219 <146 197> 195> <208 <173 208> <47 146> 173> 219> <161 161> 116> <249 <151 <104 47> <70 104> <141 70> 141> <213 <136 <11 11> <234 <83 <78 83> 249> 151> 213> 136> <12 <117 <248 <240 234> 12> 78> <17 17> <62 62> <204 248> 117> 240> 204> <24 <14 24> <46 14> 46> <228 228> <26 26> <222 222> <52 52> <182 182> <131 <189 131> <72 189> 72> <111 111> <123 <236 123> 236> <12 12> <70 <124 70> 124> <146 <179 179> 146> <85 85> <218 <220 218> <193 220> 193> <187 187> <57 <134 57> <208 134> <235 208> <122 <177 177> <151 <20 151> 122> <103 20> <164 <54 54> 235> 103> 164> <224 224> <146 146> <144 144> <122 122> <95 <212 212> <204 95> 204> <20 <144 20> 144> <108 <116 116> 108> <35 35> <45 <5 45> 5> <248 <247 248> 247> <12 <245 12> <226 <94 226> 245> 94> <233 <244 233> 244> <231 231> <137 137> <140 <227 227> 140> <211 211> <90 90> <111 <165 111> <104 104> 165> <216 <137 137> 216> <112 <8 112> 8> <226 226> <157 157> <50 <178 <60 <57 178> <5 <205 50> 205> <174 60> <248 248> 174> 57> 5> <200 <218 200> 218> <87 87> <180 180> <169 169> <248 248> <132 <116 116> 132>"

#pagebreak()

An example world:

#box(stroke: 1pt + blue, radius: 5pt, inset: 10pt)[
#for value in s.split() {
    text(raw(value + " "), fill: get_color(value), weight: 400, size: 6pt)
}
]


=== `flat_shuffle`
The languages described above are each based on a single rule. While such simplicity certainly makes analysis easier, I hypothesized that adding more complexity into the data can improve model performance.

I decided to use an idea of `Shuffle` languages from @chiang2022transferability as an extra pattern, because it was orthogonal to the bracket balancing essence of the previous datasets. The combined dataset is based on `flat`, but each consecutive group of $16$ tokens has a range of $8$ bracket types assigned to it, and all brackets on this segment are sampled only from these types. That is, each such group is a permutation of the corresponding brackets.

It adds two interesting dynamics to the task of next token prediction. When in the middle of the sequence, the model needs to look at previous tokens to guess the range of bracket types. And when close to the end of permutation, the model can guess increasingly more accurately by remembering which tokens were already used. In particular, the last token in each permutation can be predicted with certainty. Surprisingly, even small Transformer models were able to capture this pattern and indeed predicted the last token with close to zero loss.

#pagebreak()
Generation algorithm in Python:
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

#let s = "<58 <61 61> <54 58> 54> <57 57> <55 <56 <60 56> <59 55> 60> 59> <120 <121 120> 121> <125 <124 124> 125> <127 127> <123 123> <122 <126 126> 122> <147 147> <153 153> <148 148> <150 150> <151 151> <149 <152 152> 149> <154 <79 154> 79> <74 74> <75 75> <76 76> <81 <80 81> 80> <78 78> <77 77> <23 23> <21 21> <24 24> <26 26> <28 28> <25 <27 27> <22 22> <31 25> 31> <25 25> <26 <30 <29 <27 27> <32 32> <28 <127 30> 127> <130 130> 29> <134 134> <132 26> 132> 28> <131 <129 129> <128 <133 131> 133> <77 <75 77> 128> <78 <74 78> 75> 74> <80 <79 80> 79> <76 <81 <87 <82 <81 <80 80> 87> 76> 82> <84 81> 81> <85 84> <86 <83 86> 83> <32 <30 32> <34 30> 34> <29 85> <31 31> <33 29> <35 33> <36 <50 36> 35> <54 <51 <57 54> 51> 50> 57> <53 53> <55 55> <56 <52 56> 52> <166 166> <171 171> <170 170> <165 <169 <167 169> <168 167> 168> 165> <164 164> <194 194> <195 <190 <189 189> <196 <192 <193 195> 193> 190> 192> <191 196> 191> <110 110> <104 <106 106> <103 <107 <105 103> 104> <109 107> 105> <108 <128 109> 128> 108> <131 <127 131> 127> <132 132> <126 126> <125 125> <130 130> <129 <231 <236 129> <229 <232 231> <230 <233 230> <235 236> 229> 233> 235> 232> <234 <222 222> <218 218> 234> <223 223> <216 216> <217 217> <221 221> <220 220> <219 219> <6 6> <5 5> <8 <10 8> <12 <9 10> <11 11> 9> 12> <7 <110 7> 110> <115 115> <112 <114 112> 114> <111 111> <116 116> <109 109> <113 113> <101 101> <99 99> <102 102> <100 100> <105 105> <104 <98 <103 <42 <40 98> <39 <38 103> <41 <37 <43 104> <44 42> 44> 41> 40> 37> 38> 43> 39> <144 144> <150 150> <147 147> <146 146> <151 151> <149 149> <145 145> <148 <17 <21 17> 148> <15 21> <14 15> <20 <19 <18 20> <16 <33 14> <31 33> <37 <35 <36 19> 36> 37> <32 <34 <38 <159 <152 <158 <157 <155 <156 32> 35> <154 155> 159> <153 <98 157> <95 95> 31> 158> <99 <97 98> 154> <96 96> <100 <93 <94 94> <199 100> 18> 38> 199> 99> <204 97> 156> 204> <200 153> 93> 34> 152> <201 201> 200> 16> <198 198> <202 <203 202> 203> <197 197> <235 <229 <233 229> 233> 235> <232 232> <230 <234 <236 234> 230> <231 231> <91 91> 236> <95 <92 95> <94 <90 90> 92> <93 <96 96> 94> 93> <89 89> <186 186> <187 187> <185 185> <188 188> <192 192> <189 189> <191 191> <190 190> <115 115> <114 114> <112 112> <118 <111 111> 118> <116 116> <117 117> <113 113> <141 141> <142 142> <136 136> <140 140> <135 <138 135> <137 137> <139 139> 138>"


Example word:
#box(stroke: 1pt + blue, radius: 5pt, inset: 10pt)[
#for value in s.split() {
    text(raw(value + " "), fill: get_color(value), weight: 400, size: 6pt)
}
]

#pagebreak()
== Hierarchy of language complexity

=== Methodology

Some languages, both synthethic and naturals, are more complex than others. For example, it is much easier to understand the concept of balanced bracket sequences than to learn Chinese. Moreover, some languges can be understood easier if the learned already knows another language, e.g. humans need less effort to switch to a language from the same language family and large language models can be fine-tuned to a similar downstream task by using much less data than was used for their pre-training. But how to formalize this notion of complexity and similary, and how to measure it?

One approach is the Chomsky hierarchy of languages @chomsky1956three. It formally defines several classes of grammars, each one strictly more general than the previous one, and the properties of these classes are very well understood. It is very useful as a high-level formalization of language complexity, which can show that one language is qualitatively more complex than another. For example, `nested` is a context-free language, while `flat` is context-dependent. But for languages from the same class we need some other tool to find more fine-grained differences.

As the main topic of this work is transfer learning, a natural approach is to use transfer learning metrics to estimate both the complexity and similarity of languages. For a given language pair I pre-train a language model on the first language and then fine-tune it on the second language, in several stages. On the first stage I freeze all model parameters except for embeddings and unembeddings, so that the internal computations are preserved as much as possible. On the second stage the affine parameters of LayerNorms are also fine-tuned, which allows the model to re-scale the intermediate activations and thus focus on different features. Finally, the last Transformer block is fine-tuned, which allows some amount of arbitrary task-specific computation. I don't consider fine-tuning all parameters, because in the limit it can reach the same performance as the model trained on the second language from scratch and it will be not informative.

An important obseration is that transfer learning between languages is not symmetric, and it allows to estimate both (relative) complexity and similarity of two languages. If languages are similar, fine-tuning should go well in both directions, so we can take average difficulty of fine-tuning from both directions as a measure of similarity. And if one language is more complex than another, at least in a sense of having strictly more patterns, one would expect fine-tuning to run much easier from the hard language to the easy one, so comparison of two directions allows us to reason about hierarchy of languages in terms of complexity.

This procedure is not completely formalized, in particular, I don't have a principled way to measure the "difficulty" of fine-tuning. An approach that seems reasonable is to look at the first stage of fine-tuning which is enough to get the performance close to the performance of the model pre-trained on the second language. 

=== Results
#note[Why TinyStories-8M? What hyperparameters?]

In the table below there are the results of fine-tuning in both directions on certain pairs of languages. Columns "L2 ..." describe fine-tuning on the second language after pre-training on the first one, "L2 full" is the performance of the model trained on the second language from scratch. Columns "L1 ..." and "L1 full" are symmetrical, for fine-tuning on the first language. "E", "L" and "T" mean what layers were fine-tuned and stand for embeddings, LayerNorms and (the last) Transformer block. I use the absolute difference of 0.2 nats per token as a threshold for "close performance". 



#let finetuning = csv("finetuning.csv")
#set text(size: 7pt)

#align(center)[#table(
    columns: (60pt, 60pt) + (23pt,) * 8,
    fill: (col, _) => if (col < 2 or col == 5 or col == 9) { luma(240) } else { white },
    align: center,
    stroke: 0.3pt,
    [*L1*], [*L2*], [*L2 E*], [*L2 EL*], [*L2 ELT*], [*L2 full*], [*L1 E*], [*L1 EL*], [*L1 ELT*], [*L1 full*],
[nested],[flat],[4.41],[4.14],[4.08],[3.78],[*3.47*],[*3.34*],[*3.34*],[3.32],
[flat],[flat_shuffle],[2.46],[2.36],[*2.15*],[2.00],[*3.82*],[*3.80*],[*3.76*],[3.78],
[flat_shuffle],[english],[2.42],[2.30],[2.00],[1.19],[2.77],[2.62],[*2.11*],[2.00],
[nested],[english],[2.82],[2.65],[2.37],[1.19],[3.83],[*3.46*],[*3.32*],[3.32],
[flat],[english],[2.74],[2.56],[2.35],[1.19],[4.28],[4.16],[*3.76*],[3.78],
)]

#set text(size: small-size)
The first two rows show that `flat` is more complex than `nested` and `flat_shuffle` is more complex than `flat`, in a sense of being more general, because fine-tuning in the direction `flat_shuffle` #sym.arrow `flat` #sym.arrow `nested` achieves good performance simply by replacing the embeddings.
The remaining measurements showo that English is more complex than all synthetic languages used here, but it is also quite different, as the models needs more flexibility to adapt from English to e.g. `flat` and `flat_shuffle`.

=== Mechanistic interpretation
As discussed before, `nested` is a context-free language while `flat` is context-dependent. However, the fact that they lie in different classes does not explain, for example, why the second one leads to better structured embedding space. So, to have a better chance of understanding the complexity of language and its structure, reasoning in terms of abstract classes of languages and other high-level generalizations is not enough, and one should understand the actual algorithm implemented by the language model trained on it.

My first step in this investigation was to check the importance of different layers on the end result. The model considered was the one trained on `nested`, because it is both simple and very structured language, so it is easy to detect whether the model works or not. I took the average loss on several prompts as a metric and then tried zeroing out every parameter tensor in the model one at a time, comparing their impact on the performance. The result was that the most impactful layers are the first and the last one, and also embeddings and unembeddings, but the middle ones still had a nontrivial impact. In other words, I wasn't able to reduce the model to a simpler one.

As a less agressive technique, I tried replacing each layer by its low-rank approximation. Again, the middle layers were the easiest to compress, rank $4$ (out of $d_"model" = 256$) approximation of layers $2 - 5$ still recovered most of the performance. But even for rank $4$ matrices it is not trivial to understand what algorithm is implemented by them, so this approach was not very productive as well.

Then I tested the model on prompts like `<1 <2 <3 ... <n n> ... >3 >2 >1` and looked for the maximum $n$ for which the next token has the highest predicted probability at every position with a closing bracket. I found that the maximum $n$ was $16$, which combined with the fact that the model has $8$ layers allows to reject the hypothesis that each layer increases maximum nesting depth by $1$. At the same time, the model clearly had not learnt the general algorithm, otherwise it would work for any $n$, or at least much larger ones.

These observations inspired me to think about algorithms that use arithmetic in vector spaces to approximate a stack. In particular, I made the following algorithm, which predicts the token on top of the stack in a vectorized fashion: 

```python
import torch as t

n_types = 16
dim = 16
factor = 2.0

type_embedding_matrix = t.randn((n_types, dim))
type_embedding_matrix /= type_embedding_matrix.norm(dim=1, keepdim=True)

n_pairs = 16
is_open = t.tensor([1] * n_pairs + [0] * n_pairs)
bracket_type = t.tensor(list(range(n_pairs)) + list(range(n_pairs - 1, -1, -1)))

elevation = t.cumsum(2 * is_open - 1, dim=0) - is_open
weight = t.pow(factor, elevation)
signed_weight = (2 * is_open - 1) * weight
type_embeddings = type_embedding_matrix[bracket_type]
weighted_embeddings = signed_weight.unsqueeze(-1) * type_embeddings
prefix_sums = t.cumsum(weighted_embeddings, dim=0)
top_distribution = t.softmax(prefix_sums @ type_embedding_matrix.T, dim=-1)
```

The idea is that each type of bracket has a direction associated with it, and stack is a single vector, for which the distribution of the possible element "on top of the stack" is given by dot products with the type directions. To put something on top, I just add the corresponding direction with large enough weight so that it dominates everything that was accumulated in the stack before. And to pop from the stack, I subtract the same direction.

The algorithm produces accurate preditions, putting most of the probability mass into the correct token:

#align(center)[#image("../img/mech.svg", height: 250pt)]

Checking whether the language model indeed works in this way remains an open question, however it seems plausible that it implements at least an approximation of this, e.g. exponentiation is a hard operation for a language model so it might instead learn a piecewise-linear function to approximate $2^x$. 

Note that the algorithm can be trivially extended to `flat` language, by removing the exponentiation part --- now the "stack" is simply a sum of the embeddings of tokens in it. This finally provides a possible explanation of the structure of the embedding space. For `nested`, the only important property is that each vector has higher dot product with itself than with other vectors, because the embedding of the last open bracket will have more weight than all other tokens in the stack and so they will not interfere with eah other. But for `flat` the model needs access not only to the top of the stack, but to all the tokens in it, which means that arbitrary linear combination of the embeddings should be uniquely decodable. This requires the embeddings to be orthogonal. 

For all four models I measured whether their embedding vectors have the same, or close, norm, and whether they are close to orthogonal. The results are shown in the table below. Covariance means the average dot product of two different embedding vectors, and variance means the average squared norm of an embedding vector.

#align(center)[
#table(
    columns: (60pt,) + (40pt,) * 4,
    align: (left,) + (center,) * 4,
    stroke: 0.3pt,

[],[*nested*],[*flat*],[*shuffle*],[*scratch*],
[norm mean],[3.60],[3.70],[4.05],[0.79],
[norm std],[1.31],[1.13],[1.23],[0.29],
[covariance],[0.65],[0.19],[0.06],[0.16],
[variance],[14.68],[14.97],[17.94],[0.72],
)]

The ratio of the covariance to the variance is $4$ times smaller for `flat` and even smaller for `shuffle`, which supports the explanation above.


== Word-level information
There are non-trivial performance gains in all language pairs from simply tuning the embeddings, so in this section I am going to analyze the structure of the embedding space and what information about words can be extracted from the embeddings.

=== Dimensionality and clusters

The embedding dimension of the model used is $d = 256$, and human intuition, as well as many visualization techniques, works poorly for $256$-dimensional vectors, so I employ two quantatitive approaches.

First, for a $n times d$ matrix of embeddings $E$, I consider its singular values (after zeroing out the mean of each column), or equivalently, the spectrum of the covariation matrix $A = E^T E$. The motivation behind this is that if all embeddings were contained in a $k$-dimensional subpace, and $E$ had a rank $k$, then only $k$ of the singular values would be nonzero. For real data it is not the case, all singular values are nonzero due, but still some directions have much larger variance then others and the model is more likely to use features corresponding to those dimensions. 

As we see in the figure below, in models pre-trained on synthetic datasets the spectrum is dominated by the first few dimensions. In particular, before the fine-tuning, most of the interesting information about brackets is described by two axes: open-close and low-to-high bracket type id. And while they learn more diverse features during fine-tuning on English, as described in the next sections, they still don't use the embedding space very efficiently. But the interesting observation is how the tail of the spectrum behaves for models trained on different datasets: spectrum of `flat` decays to zero slower than the one of `nested`, but the shape is similar, while the spectrum of `shuffle` crosses `flat` at some point and behaves more similar to the spectrum of the model trained on English from scratch. 

#align(center)[#image("../img/spectrum.svg", height: 250pt)]

Another interesting property is how the embeddings are clustered. To quantify it, I run KMeans clustering for the embeddings varying the number of clusters and compare the plots of unexplained variance (inertia). Again, after pre-training on a synthethic language the models have only two clusters: open and close brackets, and even after fine-tuning the first few splits explain the majority of variance. But looking at the tail behaviour we observe a similar pattern: English is followed by `flat_shuffle`, then by `flat` and then by `nested`.

#align(center)[#image("../img/clusters.svg", height: 250pt)]

#pagebreak()
=== Linear probes for word features

Now that we know something about the structure of the embedding space, a natural question to ask is how this structure is used. In other words, what information about a word can one extract from the embedding of the corresponding token.

Preliminary experiments showed that clusters of features correspond to properties like "noun", "3rd person verb", "adjective or adverb", etc. Therefore I decided to use part-of-speech tags provided by NLTK library as targets. Initially there were more than $30$ unique tags in the dataset, but many of them were very rare. After filtering out all tags with less than $200$ occurences the following tags remained:
- CD: cardinal digit
- IN: preposition or subordinating conjunction
- JJ: adjective
- NN: singular noun
- NNP: proper noun
- NNS: plural noun
- RB: adverb
- VB: base form verb
- VBD: past tense verb
- VBG: gerund
- VBN: past participle

I added a feature indicating the frequency of the token in the training corpus, because typically the direction with the most variance in the embedding space roughly corresponded to frequency. And the last feature is whether the token starts with a whitespace, as some clusters had disproportionate amount of such tokens.

For each of the models and each of the features I trained a ridge regression (for frequency) or a logistic regression (for all other variables, as they are boolean) on $80%$ of the embeddings and then evaluated their $R^2$ score or ROC-AUC on the remaining $20%$.


#align(center)[#table(
    columns: (60pt,) + (40pt,) * 4,
    fill: (col, row) => if (col == 0 or row == 0) { luma(255) } else { white },
    align: (left,) + (center,) * 4,
    stroke: 0.3pt,

[],[*nested*],[*flat*],[*shuffle*],[*scratch*],
[`frequency`],[0.84],[0.85],[0.85],[0.93],
[`start_space`],[0.70],[0.70],[0.70],[0.89],
[`pos_tag_CD`],[0.66],[0.63],[0.63],[0.80],
[`pos_tag_IN`],[0.76],[0.79],[0.71],[0.87],
[`pos_tag_JJ`],[0.60],[0.58],[0.60],[0.73],
[`pos_tag_NN`],[0.63],[0.62],[0.63],[0.76],
[`pos_tag_NNP`],[0.64],[0.65],[0.63],[0.79],
[`pos_tag_NNS`],[0.67],[0.67],[0.68],[0.84],
[`pos_tag_RB`],[0.69],[0.63],[0.64],[0.84],
[`pos_tag_VB`],[0.71],[0.69],[0.68],[0.79],
[`pos_tag_VBD`],[0.75],[0.71],[0.67],[0.89],
[`pos_tag_VBG`],[0.71],[0.70],[0.73],[0.89],
[`pos_tag_VBN`],[0.72],[0.68],[0.72],[0.87],
[*Average*],[0.70],[0.68],[0.68],[0.84],
)]

All probes in for all models perform better than random, so every model learns at least something related to this features. The embeddings of the model trained on English from scratch predictably outperformed the others, but the quality of other embeddings turned out to be on average the same. Perhpaps the difference in effective dimension between the models is used not for this relatively simple single word features, but for word meanings, sentence structure and so on.

== Cloze tests

To assess how good the models are in understanding language in general a different benchmark is needed. As the models studied are too small for reliable question answering, reasoning and other high-level cognitive skills, the test should be as simple as possible, ideally just measuring perplexity on some texts. There are several already existing datasets for natural language understanding, such as GLUE @wang2018glue and MMLU @hendrycks2020measuring, and they have many diverse subtasks, but they focus on more complex tasks.

Instead, I used GPT-4 @openai2023gpt4 to generate a set of cloze infilling questions in simple English. There are the following $12$ subtasks, each with $10$ cloze questions:
- Synonyms and antonyms.
- Logical relations. 
- Subject-verb agreement.
- Prepositions.
- Conjunctions.
- Temporal understanding.
- Spatial understanding.
- Quantitative reasoning.
- Emotions.
- Narrative understanding.
- Ethics.

Each cloze question consists of a prompt with a cloze marker, a correct answer and an incorrect answer. For each question the difference between log-probabilities of the correct and incorrect answers is measured and then averaged across each subtask.

An example from the temporal understanding subtask:

#box(stroke: 1pt + blue, radius: 5pt, inset: 10pt)[
```json
[
    "She ate breakfast # she went to school",
    "before",
    "after",
]
```
]

For each of the synthetic languages I used two models, one where only the fine-tunings were adapted to English (E) and another with all three stages (ELT). I compared them to the model of the same architecture trained on English from scratch, and also to a four times bigger model trained on English from scratch to see which metrics can be improved.
#pagebreak()

#align(center)[#table(
    columns: (90pt,) + (33pt,) * 8,
    fill: (col, row) => if (col == 0 or row == 0) { luma(255) } else { white },
    align: (left,) + (center,) * 8,
    stroke: 0.3pt,
[],[*nested ELT*],[*nested E*],[*flat ELT*],[*flat #h(0.3em)  E*],[*shuffle ELT*],[*shuffle E*],[*scratch 8M*],[*scratch 33M*],
[#text(`synonyms and antonyms`, 7pt)],[0.18],[0.24],[0.13],[0.22],[0.25],[0.31],[0.25],[0.28],
[#text(`single plural`, 7pt)],[0.15],[0.08],[0.50],[0.19],[0.33],[0.03],[0.58],[0.71],
[#text(`logical relations`, 7pt)],[-0.30],[-0.08],[-0.18],[-0.44],[-0.08],[-0.13],[-0.04],[0.09],
[#text(`subject verb agreement`, 7pt)],[0.45],[0.54],[0.36],[0.46],[0.26],[0.14],[0.83],[0.98],
[#text(`prepositions`, 7pt)],[0.52],[0.43],[0.53],[0.51],[0.48],[0.40],[0.94],[1.12],
[#text(`conjunctions`, 7pt)],[0.43],[0.46],[0.38],[0.45],[0.49],[0.36],[0.63],[0.82],
[#text(`temporal understanding`, 7pt)],[-0.02],[-0.13],[0.04],[-0.14],[0.36],[0.09],[0.44],[0.73],
[#text(`spatial understanding`, 7pt)],[0.30],[0.13],[0.48],[0.40],[0.37],[0.06],[0.64],[0.71],
[#text(`quantitative reasoning`, 7pt)],[0.00],[-0.06],[-0.01],[-0.14],[-0.04],[-0.14],[-0.04],[-0.06],
[#text(`emotions`, 7pt)],[0.03],[-0.08],[0.07],[0.05],[-0.01],[0.20],[0.61],[0.77],
[#text(`narrative understanding`, 7pt)],[-0.04],[-0.07],[0.07],[0.03],[0.04],[0.04],[0.17],[0.27],
[#text(`ethics`, 7pt)],[0.17],[0.32],[0.22],[0.34],[0.30],[0.27],[0.25],[0.51],
[*Average*],[0.16],[0.15],[0.22],[0.16],[0.23],[0.14],[0.44],[0.58],
)]

In terms of general trends there are two interesting observations. First, models with all three stages of fine-tuning are better, predictably, than their counterparts having only the embeddings tuned, but this difference is more pronounced in `flat` and `flat_shuffle`. The question why is it so remain open for the future work. Second, a familiar pattern appears again, `nested` < `flat` < `flat_shuffle` < `scratch`, which proves the superiority of the introduced `flat_shuffle` dataset.


= Conclusion

#note[Say something about hypotheses, e.g.:
- Complex synthethic languages lead to better results, as demonstrated by embedding structure and cloze test performance of `flat_shuffle`
- Models are not strictly limited by the complexity of the synthethic dataset and can learn features that don't have direct analogy in the pre-training task 
- Perhaps a more sophisticated pre-training dataset simply causes the model to have richer structure, which then can be used in completely different ways during transfer learning, as in reservoir computing
]