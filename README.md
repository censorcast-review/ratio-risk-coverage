# Accuracy and Coverage under Ratio Risk

Reproducibility source and manuscript assets for **When More Predictions Cover
Less: Accuracy and Coverage under Ratio Risk**.

- [Manuscript PDF](paper/CENSORCAST_ICLR_2027_Final.pdf)
- [Manuscript source](paper/main.tex)
- [Repository](https://github.com/censorcast-review/ratio-risk-coverage)

## Contents

The release includes the integrated manuscript, its figures and tables, the
three-seed M5 refit code, the Dominick's oatmeal external test, the conditional
perturbation-bound checks, and the Bibtex/Mediamill experiments and learned-gate
comparison. Earlier scientific code and frozen protocols remain at their
original relative paths because the reproducibility pipeline refers to them.
Historical manuscript copies and editorial records are not part of this release.

## Build the paper

Use Python 3 with `pypdf`, and a LaTeX installation providing `pdflatex`,
`bibtex`, and the packages used by the manuscript. From the repository root:

```sh
python code/verify_release.py
python code/build_paper.py
```

The build writes to `reproduction_outputs/paper/`, checks the nine-page main-text
limit, and rejects overfull boxes and unresolved citations or references. Its
manifest check verifies the files distributed here; it is not an independent
verification of the experimental findings.

## Reproduce numerical results

**The source checkout does not contain the numerical replay arrays or fitted
models.** Numerical replay requires the separately supplied
`CENSORCAST_ICLR_2027_Supplement_v13.zip`. Read the attachment's `README.md` for
the compact replay commands and dependency details before overlaying its
`CENSORCAST/` contents onto the checkout. Preserve this release's root
`README.md`, `MANIFEST.json`, and `RELEASE_SELECTION.json` during the overlay.
The compact supplement verifies the reported fixed-policy statistics; it does
not contain every calibration row or fitted model needed for full retraining.

The execution dependencies are recorded in `requirements.txt`,
`requirements-matched.txt`, `requirements-v12.txt`, and
`evidence/revision_v13_nonwape/requirements-v13.txt`. Full retraining additionally
requires the original datasets, appropriate dataset access, and the larger
experiment inputs described by the protocols. Raw Dominick's and Mulan datasets
are not redistributed here. No dataset license is replaced by this release.

Frozen `PROTOCOL.json` files and scientific scripts retain their original bytes.
Some contain paths from the original execution environment; use the supplied
restoration and replay instructions to work from a different checkout path.
Observed external outcomes include failed joint success conditions and failed
risk screens; the manuscript reports these outcomes and does not establish a
population risk guarantee.
