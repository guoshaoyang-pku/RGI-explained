# RGI-explained

## Commands

    ./setup.sh
    python3 verify_evidence.py
    python3 -m http.server 8000
    latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error paper/main.tex

The basic verifier uses the Python standard library. XeLaTeX/latexmk are optional paper-build prerequisites. Scientific producer/checker/plot sources require their original NumPy, PyTorch/CUDA, or Matplotlib environment and omitted data/weights/logit fixtures.

## Files

- paper/main.tex and paper/sections/: MetaCircle manuscript; Shaoyang Guo sole core author, Ziming Liu corresponding only.
- paper/figures/: four-term plots and measured tables.
- paper/main.pdf, paper/rgi-followup-arxiv.zip, and paper/source-manifest.json: PDF, independently compiled source archive, and hashes.
- evidence/common-{initial,refinement,local}/: three retained common-start rate protocols, all 54 traces and compact scalar records.
- evidence/common-claims.json and common-source-provenance.json: current claims and transformations.
- evidence/experiment-summary.md: current protocol and evidence availability.
- experiments/common_checkpoint/: recorded producer, advanced checker, plotting code, and contracts.
- docs/index.html: bilingual native-HTML visual abstract with normal-size fonts.
- docs/reports/rgi-followup.md: current Chinese summary.
- docs/reports/rgi-followup-panel.html and older docs/assets/visual-abstract-* exports: historical distinct-start diagnostics.
- index.html and .nojekyll: static GitHub Pages entry.
- RELEASE_MANIFEST.json: exact intended release file hashes.

## Scientific boundaries

The main experiment starts every branch from the same official step-143000 checkpoint with reset optimizer state, shared batches, and 32-step warmup. All three protocols reuse the same pool with three paired permutations. Later rate protocols are adaptive; retain earlier failures. Historical experiments have different optimizer-specific starts and must stay labeled.

Preserve the exact identity M=L−D=W+Q=A+R+Q and the measured approximation A+Q₂. ε=R+(Q−Q₂), and network R can be second order. The head-only intervention is affine in updated parameters. D's coefficient one is an identity, not evidence for equal optimizer dynamics. Displacements are realized; these tests do not forecast unknown future updates.

Keep the failed full-window SGD/HB response, paired-response, and strong token constant-gap tests. Endpoint parameter cosine is not whole-trajectory alignment or token-wise RGI. A fixed checkpoint is not the natural Bayes distribution. The release and original checker do not replay every gradient/training update.

## Editing

Build after LaTeX edits. Recompute figures/tables from included evidence rather than editing numbers by hand. Regenerate RELEASE_MANIFEST.json after all intended files settle. Keep private paths, secrets, local audit reports, and logs outside the committed tree. Use explicit staging in the shared research workspace; the public checkout is dedicated to this release.
