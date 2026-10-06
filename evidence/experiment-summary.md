# Experiment evidence summary

## Scientific scope

This package reports a finite local continuation of Pythia-70M. It is not a reproduction of the prior RGI paper's 124M/0.7B from-scratch experiments, and it does not claim to recover the historical Pythia AdamW/Pile optimizer state. Natural targets are scored with teacher forcing. A fixed checkpoint reference is a measurement reference; its KL is not called natural Bayes excess risk.

## Data and protocol

- FineWeb sample-10BT, revision 9bb295ddab0e05d785b879661af7260fed5140fc, shard sample/10BT/001_00000.parquet.
- Source rows begin at 65,536; 4,672 unique length-qualified documents; one 256-token span per document.
- Split 4,096 train, 64 calibration, 512 evaluation; three permutations reuse the same pool.
- Prior registered IDs, text hashes, and exact spans were excluded. Selection used no model score. Absence from Pythia pretraining is unknown.
- Natural continuation: 512 updates, batch 8, context 256; SGD learning rate 0.001; Adam learning rate 1e-6, betas (0.9, 0.95), epsilon 1e-8; optimizer state reset at continuation start.
- Shared reference: official Pythia-70M step 32,000; each natural algorithm also has its own frozen start.
- Known teacher: official step 32,000, 33 natural seed tokens plus 63 full-vocabulary temperature-1 samples, vocabulary 50,304; 1,024/64/128 train/calibration/evaluation; student starts at official step 16,000; 128 updates, hard sampled or soft teacher objective.

## Main verified numbers

Frozen difficulty: common step-32,000 reference explains 96.657-97.578% near and 96.745-97.582% far raw variance. Algorithm-specific frozen starts explain 99.9943-99.9974% near and 99.9400-99.9713% far; RMSE is 0.000916-0.001330 nats near and 0.003663-0.004736 far. Batch difficulty standard deviation is approximately 0.1761-0.2165 nats.

Local slow response: for M=L-D, A+Q2 has raw relative RMS 4.97-9.73% for SGD and 1.19-2.82% for Adam across near/far/full windows, with variance explained at least 96.87%. Full-window means are SGD M=-0.002153..-0.001892, A=-0.004011..-0.003552, Q=+0.001740..+0.001981; Adam M=-0.002699..-0.002690, A=-0.003290..-0.003272, Q=+0.000609..+0.000621 (nats). Signed low-frequency shares show work above one and curvature negative because the terms cancel.

Boundaries: scalar probe relative RMS is 45.1-90.0% near, 29.5-55.2% far, 31.7-62.8% full. SGD far high-pass is 10.75-16.68%. Delta A + Delta Q2 for optimizer response is 10.34-16.68% full raw error in all three replicas, despite 96.74-98.57% variance explained.

Known teacher: teacher entropy mean 4.1798397 nats; sampled teacher surprisal mean 4.1723279; initial teacher KL 0.2593348. Final teacher KL ranges: hard SGD 0.229442-0.229879; hard Adam 0.241563-0.241657; soft SGD 0.228137-0.228251; soft Adam 0.229840-0.229902. Population A+Q2 response error is at most 2.795%. Strong constant-offset prefix std/absolute mean is 3.59-3.69 hard and 10.69-11.89 soft, failing the 0.1 gate.

## Evidence boundaries and verification

L-D=W+Q is exact arithmetic. A+Q2 uses the observed parameter/logit displacement and is a finite-response explanation, not an unknown-future update predictor. The extension, logit-work, and arrival verifiers report 23,220, 4,159, and 341 passed checks. Maximum reported arithmetic errors are 8.88e-15, 3.17e-14, and 3.56e-15. These checks do not independently recompute full network gradients, complete optimizer trajectories, or every saved logit from raw checkpoints.

Public aggregate analyzer outputs are included in evidence/numerical-theory-summary.json, evidence/arrival-analysis.json, evidence/work-analysis.json, and evidence/extension-analysis-fullpair-separability.json. Figure values are in evidence/numerical-theory-values.json. Evidence/source-provenance.json identifies original artifacts and transformations by hash. Raw checkpoints, full producer traces, logits, training data text, and omitted notes are not included. This package supports inspection of reported aggregates and chart values; it is not a complete training reproduction package.
