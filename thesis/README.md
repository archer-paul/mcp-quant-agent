# Thesis LaTeX source

Source for the MSc thesis *Autonomous Trading Agents via the Model Context Protocol*.
Self-contained: this `thesis/` directory compiles on its own.

## Compile order

This project uses `biblatex` with the **biber** backend, plus `cleveref`/`hyperref`, so a
single `pdflatex` pass is not enough. Run, from inside `thesis/`:

```
pdflatex main
biber    main
pdflatex main
pdflatex main
```

(Equivalently `latexmk -pdf main.tex`, which sequences this automatically.)

> I do not compile for you — run it yourself. No `pdflatex` is invoked by the agent.

## Where the numbers live

**All result numbers are centralised in `results/numbers.tex`.** Nothing in the chapter
sources hard-codes a figure; the body cites macros such as `\faithBull`, `\agentSharpe`,
`\groundingOverall`. Until the full run lands, each macro expands to `\TODO{...}`, which
renders as a red **[TODO: ...]** flag in the PDF — so no placeholder can slip through
unnoticed.

To finalise: open `results/numbers.tex` and replace the `\TODO{...}` on the right-hand side
of each `\newcommand` with the value from the run artifact. **Do not edit the chapter
`.tex` files** — they reference the macros only. A trailing comment on each macro line is
reserved for the preliminary medium-scale value (author convenience; not used by the build).

Hard rule encoded in the macros: **never annualise a Sharpe computed on a short window**
(e.g. the 19-day medium window). Any such quantity stays `\TODO` and must be filled only
from a window long enough to annualise meaningfully.

## File layout

```
thesis/
  main.tex              Master file; \input order = chapter order
  preamble.tex          Packages, macros (\TODO, \tnow, ...), styles
  references.bib        biblatex/biber bibliography
  results/numbers.tex   *** all result-number macros (mostly \TODO) ***
  frontmatter/          abstract, declaration, glossary of acronyms
  chapters/             introduction, related_work, methodology, architecture,
                        evaluation, results, discussion, conclusion, appendix
  figures/              5 hand-drawn TikZ figures (no external images required)
```

## Figures

All five core diagrams are TikZ (no external image dependencies): the clock-centred agent
loop, the layered architecture, the clock-gated data flow, the PM multi-agent pipeline, and
the evaluation protocol. One screenshot is requested in `chapters/appendix.tex` (a Langfuse
PM span tree) and is marked `\TODO` until provided.

## Writing status

See the project `HANDOVER.md` (repo root) for the per-chapter writing status: what is stable
and written, what awaits the full-run numbers, and which captures are requested.
