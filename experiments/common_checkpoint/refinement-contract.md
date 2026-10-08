# Prospective local-scale refinement

Registered after the common-start512-step replica0 SGD-low and momentum-low completed on2026-10-08, before these refinement jobs. Prior six-setting contract and all results remain unchanged.

The original nominal low h=.0003 gives good frozen-difficulty prediction (~97.2/98.2%) but fails response closure (~129/200% relative RMS), with Q2~.09nats. Thus it is not established as a local-step regime for this late checkpoint on FineWeb. This is substantive negative evidence, not a discard rule.

Test the SAME checkpoint/data/splits/masks/FP64/warmup32/permutation seeds on h=.000003 and h=.00003 (100x/10x below the previous nominal low), plus reset Adam calibrated at the smaller h, and head-only at h=.000003. Keep the worker's labels low/high purely relative rate labels and publish actual rates. Repeat all six settings and all three permutations without selecting a favorable permutation.512updates. Adam calibration again uses only the same16 calibration documents, matching the smaller SGD one-step Q2; no evaluation scores. No changes to diagnostic gates or windows.

This is an adaptive experimental extension, not an untouched prospective confirmation of the original mechanism. Checkpoint late143000 was fixed prior to either protocol. Show both protocols' four-term curves/table and response/paired gates. Learning-rate sensitivity is measured at the same update count and separately on the prespecified cumulative-rate coordinate; it does NOT prove the fixed-effective-progress asymptotic bound. Additional head-only control tests R=0.

Use idle explicitly owned H200 GPUs after live preflight, persistent /data4 storage, existing torch2.9.1 environment; no interference with GPU0/3/7 applications. Canonical tensor content hash verifies the existing remote step143000 checkpoint matches the local official file. Record host/environment difference. Common D and identities must be checked within this host and all FP64 token fixtures revalidated; cross-host exact bit equality is not assumed.
