# Beyond Relative Generalization Invariance

**Shared Difficulty and Local Learning Response in Language Model Training**

Shaoyang Guo and Ziming Liu are the core contributors. Ziming Liu is the corresponding author.

This release contains the MetaCircle manuscript, Chinese report, interactive panel and bounded aggregate evidence. It studies a finite teacher-forced Pythia-70M continuation with SGD and Adam.

## Result and scope

A common official step-32000 reference explains 96.657–97.578% of near-window and 96.745–97.582% of far-window raw arriving-loss variance. After subtracting each algorithm frozen start, observed-displacement A+Q₂ has 4.97–9.73% relative RMS error for SGD and 1.19–2.82% for Adam.

The response analysis is retrospective. A+Q₂ uses the displacement that actually occurred. The experiments do not establish universal strong RGI or equal first-order dynamics coefficients across algorithms. Scalar-probe, optimizer-difference and known-teacher constant-offset failures are retained.

## Read

- [Manuscript PDF](main.pdf)
- [Chinese report](docs/reports/rgi-followup.md)
- [Interactive panel](docs/reports/rgi-followup-panel.html): open in a browser, switch real trajectories, and export CSV.
- [Numerical summary](evidence/numerical-theory-summary.json)
- [Evidence scope](evidence/experiment-summary.md)

## Build and check

Prerequisites: Python 3.9+, XeLaTeX and latexmk with the packages used by metacircle.sty. No credentials or application environment variables are required.

    ./setup.sh
    python3 verify_evidence.py
    latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex

The arXiv zip includes main.bbl and a XeLaTeX compiler declaration. references.bib is included for local edits.

## Evidence availability

Included files contain aggregate arrival, local-work and paired-prefix analyses; figure values; bounded selection provenance; and verifier count/scope extracts. [source-provenance.json](evidence/source-provenance.json) records original and sanitized hashes. Claims identify whether each original source is included.

Raw checkpoints, full logits, producer traces, training data text and complete training scripts are not included. The checker verifies package integrity and recorded arithmetic; it does not rerun gradients or model trajectories. This is a manuscript and aggregate evidence release, not a complete training reproduction package.

## License and contributions

Original code and documentation use MIT; see [LICENSE](LICENSE). See [NOTICE.md](NOTICE.md) for third-party branding and citations. No blanket MIT grant is claimed for the Tsinghua or MetaCircle marks.

See [CONTRIBUTING.md](CONTRIBUTING.md). When using Codex or Claude Code, read [AGENTS.md](AGENTS.md) and preserve the evidence boundaries.
