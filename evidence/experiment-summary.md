# Common-checkpoint evidence summary

## Main protocol

All six settings start from official EleutherAI/Pythia-70M step 143000, revision de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852. Canonical tensor-content SHA256 is 455a8e2d0db4b5ec428130e269f125ace865c2323fb47fb3750f63b26859f530. The checkpoint is late Pile pretraining; continuations use FineWeb. Historical optimizer state is not recovered: every branch resets it and uses a 32-update linear warmup.

The reused FineWeb sample-10BT pool has 4,672 unique length-qualified documents and train/calibration/evaluation splits 4,096/64/512. One 256-token span is selected per document; 223 next-token targets at positions 33–255 are scored using true prefixes. Three permutations use seeds 202610081, 202610082, 202610083. Batch size is 8, each run has 512 updates, and arriving batch measurements are pre-update. A fixed 32-document probe is scored at steps 0,16,32,64,128,192,256,320,384,448,512. Computation uses FP64, with TF32 disabled.

SGD and normalized Heavy Ball (beta 0.9, raw PyTorch rate 0.1h) use two effective rates per protocol. Adam resets moments (betas 0.9/0.95, epsilon 1e-8); its rate matches the initial single-step Q₂ on 16 calibration documents, not progress, dc gain, or direction. Head-only SGD freezes every tensor except embed_out.weight; it preserves the starting function.

| Retained protocol | Smaller SGD/HB h | Larger h | Calibrated Adam η |
|---|---|---|---|
| Initial | 3e-4 | 1.2e-3 | 1.2685965885977084e-5 |
| Intermediate | 3e-6 | 3e-5 | 2.8994061505339026e-7 |
| Terminal | 3e-8 | 3e-7 | 3.02668013023587e-9 |

The intermediate and terminal protocols were adaptively registered after the preceding smaller-rate approximation failed. All protocols, settings, and permutations are retained: 54 complete runs. Fixed update counts are not fixed effective progress. Rate labels do not establish stability regimes.

## Exact terms and measured approximation

M=L−D=W+Q=A+R+Q. D is frozen batch cross-entropy. W averages (p₀−e_y)ᵀΔz. Q averages KL(p₀∥pₜ). A=g_B(θ₀)ᵀ(θₜ−θ₀). Q₂=½ mean Varₚ₀(Δz). R=W−A. The measured approximation error is ε=R+(Q−Q₂)=L−D−A−Q₂. All observed displacements are retrospective. ε is not merely R, and R generally includes second-order parameter-to-logit curvature.

The main terminal table records D batch std, A signed mean, Q₂ mean, and ε RMS. Values across three reused-pool permutations are available in common-local/stats.json and its figures/core-table.json. Ranges are not confidence intervals. D is exactly shared across settings at the same arrival/probe; it varies on arriving batches and is constant on the fixed probe.

## Results and failures

Terminal D predicts >99.9926% of raw arriving-loss variation. Full-window response error RMS(ε)/RMS(L−D) is 10.5109–14.9551% for smaller SGD, 10.5604–15.2109% for smaller HB, 0.338662–0.374779% for Adam, and 0.00241046–0.00247038% for head-only SGD. The gate is 10%. Smaller SGD/HB pass near 0:128 and fail far 384:512. Earlier, larger-displacement response failures are preserved.

Terminal full-network SGD has R RMS 2.419e-5–3.480e-5 nats, versus softmax residual RMS 1.349e-7–2.588e-7. The affine head-only control has R <=1.90666e-14 nats in both smaller-rate extensions. Initial high SGD instead has Q₂ mean 11.33–13.10 nats and softmax residual RMS 15.72–19.27, exposing probability-curvature approximation failure.

Smaller SGD/HB endpoint displacement cosine rises from 0.73536–0.75361 to 0.99107–0.99500 and 0.99970575–0.99980905 over the three protocols. This measures endpoints, not every trajectory point. Terminal paired response closure is 22.18–28.30%. Final SGD/HB token-gap std/absolute mean is 46.49–163.48, above the 0.1 gate. Zero of 45 terminal paired setting/permutation constant-gap tests passes. These finite continuations identify shared difficulty and response errors; they do not reproduce or universally falsify the original from-scratch RGI claim.

## Included artifacts and verification scope

Each common-{initial,refinement,local}/ directory includes plan, calibration, stats, parameter-alignment and saved verification JSON; all 18 runs' registration, summary, trace and fixed-probe JSON; compact per-document and final-token scalar NPZ arrays; and exported plots/tables. common-source-provenance.json records original versus sanitized hashes. Hardware/path fields have been removed while scientific values remain.

Original independent NumPy checks total 6,834 (2,278 per protocol), with maximum numerical errors 3.34843e-13, 9.59463e-14, and 2.38774e-14. They independently reconstruct final eight-token full-vocabulary logit fixtures, test recorded identities, paired D, schedules, hashes and JVP means. They do not independently replay every gradient or optimizer update. A separate CPU real-model pulse checks gradient-dot/JVP agreement to 6.25e-16 and exact logit arithmetic to 1.41e-15.

The public root checker uses the standard library to verify manifest hashes, recorded traces, schedules, common D, aggregate statistics and retained verdicts. Raw text/token spans, model weights, full-vocabulary logit fixtures and final parameter endpoints are omitted. The original advanced checker and producer sources are included for inspection but require those omitted inputs. Compact parameter-alignment metrics are recorded evidence, not an independent public endpoint recomputation.

## Historical supporting diagnostics

The older numerical-theory-summary.json, arrival-analysis.json, work-analysis.json, known-teacher aggregates, historical visual-abstract exports, and interactive panel use optimizer-specific starts with a step-32000 shared scoring reference. Their old claim registry is explicitly historical. Their response results must not be transferred to the new common late checkpoint or rate protocols. Historical source hashes are in source-provenance.json; the main claims are now in common-claims.json.
