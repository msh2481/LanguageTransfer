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

*Experiments*:
Take a single model, something like `GPT-2` or `TinyStories`. Pre-train it on different L1, then transfer to L2 and test there. Also, for synthetic languages, check what happens inside the pre-trained model, and how these mechanisms are used in transfer task.

*L1*:
- English (default GPT-2 pre-training)
- Flat dependencies: `<1 <2 <3 2> 3> 1>`
- Nested dependencies: `<1 <2 <3 3> 2> 1>`
- Shuffle: `0 3 1 2 , 8 9 7 6 , 11 8 9 10`
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
- Parameter-efficient fine-tuning
- Whole model

