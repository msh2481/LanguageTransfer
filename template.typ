#let script-size = 8pt
#let footnote-size = 8pt
#let small-size = 9pt
#let normal-size = 10pt
#let large-size = 12pt
#let title-size = 16pt

#let note(text) = block(
  fill: luma(240),
  stroke: luma(100),
  inset: 8pt,
  radius: 4pt,
  text
)

#set text(font: "Comic Mono")

// This function gets your whole document as its `body` and formats
// it as an article in the style of the American Mathematical Society.
#let ams-article(
  // The article's title.
  title: "Paper title",

  // An array of authors. For each author you can specify a name,
  // department, organization, location, and email. Everything but
  // but the name is optional.
  authors: (),

  // Your article's abstract. Can be omitted if you don't have one.
  abstract: none,

  // The article's paper size. Also affects the margins.
  paper-size: "a4",

  // The path to a bibliography file if you want to cite some external
  // works.
  bibliography-file: none,

  // The document's content.
  body,
) = {
  // Formats the author's names in a list with commas and a
  // final "and".
  let names = authors.map(author => author.name)
  let author-string = if authors.len() == 2 {
    names.join(" and ")
  } else {
    names.join(", ", last: ", and ")
  }

  // Set document metadata.
  set document(title: title, author: names)

  // Set the body font. AMS uses the LaTeX font.
  // set text(size: normal-size, font: "New Computer Modern")
  set text(size: normal-size)

  // Configure the page.
  set page(
    paper: paper-size,
    // The margins depend on the paper size.
    margin:
      (
        top: 117pt,
        left: 118pt,
        right: 119pt,
        bottom: 96pt,
      )
  )

  // Configure headings.
  set heading(numbering: "1")
  show heading: it => {
    v(10pt)
    // Create the heading numbering.
    let number = if it.numbering != none {
      set text(size: large-size)
      counter(heading).display(it.numbering)
      h(10pt, weak: true)
    }

    // Level 1 headings are centered and smallcaps.
    // The other ones are run-in.
    set text(size: large-size, weight: 400)
    smallcaps[
      #number
      #it.body
    ]
    v(10pt)
  }

  // Configure lists and links.
  set list(indent: 24pt, body-indent: 5pt)
  set enum(indent: 24pt, body-indent: 5pt)
  show link: set text(font: "New Computer Modern Mono")

  // Configure equations.
  show math.equation: set block(below: 8pt, above: 9pt)
  show math.equation: set text(weight: 400)
  // Configure citation and bibliography styles.
  set bibliography(style:"apa", title: "References")

  show figure: it => {
    show: pad.with(x: 23pt)
    set align(center)

    v(12.5pt, weak: true)

    // Display the figure's body.
    it.body

    // Display the figure's caption.
    if it.has("caption") {
      // Gap defaults to 17pt.
      v(if it.has("gap") { it.gap } else { 17pt }, weak: true)
      it.supplement
      if it.numbering != none {
        [ ]
        it.counter.display(it.numbering)
      }
      [. ]
      it.caption.body
    }

    v(15pt, weak: true)
  }

  // Theorems.
  show figure.where(kind: "theorem"): it => block(above: 11.5pt, below: 11.5pt, {
    strong({
      it.supplement
      if it.numbering != none {
        [ ]
        counter(heading).display()
        it.counter.display(it.numbering)
      }
      [.]
    })
    emph(it.body)
  })

  // Display the title and authors.
  v(35pt, weak: true)
  align(center, {
    smallcaps([#text(size: title-size, weight: 500, title)])
    v(25pt, weak: true)
    for a in authors {
      text(size: normal-size, weight: 700, a.name)
      v(10pt, weak: true)
      text(size: normal-size, weight: 500, a.uni)
    }
  })

  // Configure paragraph properties.
  // set par(first-line-indent: 1.2em, justify: true, leading: 0.58em)
  // show par: set block(spacing: 0.58em)

  // Display the abstract
  if abstract != none {
    v(20pt, weak: true)
    set text(large-size)
    show: pad.with(x: 35pt)
    align(center, smallcaps([Abstract ]))
    set text(normal-size)
    abstract
  }

  // Display the article's contents.
  v(29pt, weak: true)
  body

  // Display the bibliography, if any is given.
  if bibliography-file != none {
    show bibliography: set text(8.5pt)
    show bibliography: pad.with(x: 0.5pt)
    bibliography(bibliography-file)
  }
}

// The ASM template also provides a theorem function.
#let theorem(body, numbered: true) = figure(
  body,
  kind: "theorem",
  supplement: [Theorem],
  numbering: if numbered { "1" },
)

// And a function for a proof.
#let proof(body) = block(spacing: 11.5pt, {
  emph[Proof.]
  [ ] + body
  h(1fr)
  box(scale(160%, origin: bottom + right, sym.square.stroked))
})
