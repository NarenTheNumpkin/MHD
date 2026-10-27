# Manuscript

*MedHyperDiff: relation-aware diffusion for synthetic clinical laboratory tables*, prepared for BMC Medical Informatics and Decision Making with the Springer Nature LaTeX template (`sn-jnl` class, version 3.1, Vancouver numbered references).

| File | Contents |
|---|---|
| `manuscript.tex` | LaTeX source |
| `references.bib` | References |
| `sn-jnl.cls`, `sn-vancouver-num.bst` | Springer Nature template class and reference style |
| `manuscript.pdf` | Compiled manuscript |
| `figures/Figure1.pdf` to `figures/Figure8.pdf` | Figures |

Build from this folder:

```bash
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

## Figures

All figures are drawn with matplotlib in the style of [figures4papers](https://github.com/ChenLiu-1996/figures4papers).

```bash
python scripts/figure_overview.py paper/figures                        # Figure 1
python scripts/figures_paper.py reproducibility/figure_inputs paper/figures/build   # Figures 2 to 8
```

| Figure | Output of `figures_paper.py` |
|---|---|
| 2 | `fig_ablation.pdf` |
| 3 | `fig_reversals.pdf` |
| 4 | `fig_precision.pdf` |
| 5 | `fig_memorization.pdf` |
| 6 | `fig_evaluation_size.pdf` |
| 7 | `fig_first_test.pdf` |
| 8 | `fig_robustness.pdf` |
