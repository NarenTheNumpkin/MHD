# Figures

`figures/Figure1.pdf` to `figures/Figure8.pdf` are the figures of the manuscript.

Figure 1, the overview of the method, is drawn in TikZ following the conventions of [PetarV-/TikZ](https://github.com/PetarV-/TikZ):

```bash
cd paper/figures/src && pdflatex overview.tex
```

Figures 2 to 8 are drawn with matplotlib in the style of [figures4papers](https://github.com/ChenLiu-1996/figures4papers) from the CSV files in `reproducibility/figure_inputs/`:

```bash
python scripts/figures_paper.py reproducibility/figure_inputs paper/figures/build
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
