# Independent figure scale audit

Recomputed all 54 arrival traces, all 54 fixed-probe traces, and all 54 final-token arrays using Python standard-library arithmetic. No source evidence was changed. Population standard deviations and RMS are in nats. Replica 0 remains the display replica; every reported range below covers all three registered data permutations.

## Recommended front figure

Use the **intermediate** protocol, registered replica 0, steps **300–310 inclusive**. Draw the measured shared difficulty D and all six measured loss curves. Add a separate response or paired-gap panel so 10^-3–10^-2 effects remain visible. Call these **average setting-dependent offsets**, with sample-dependent variation. Do not claim an exactly fixed loss gap.

At replica 0 in this window, D std is 0.229261 nats. Relative to smaller-rate SGD, average gaps are: Adam −0.0168476, larger-rate SGD −0.0105775, larger-rate Heavy Ball −0.00836254, head-only +0.00241730, and smaller-rate Heavy Ball +0.000139141. Thus the requested scale fits several intermediate-protocol settings, but not every pair.

## Raw arrival scales across three permutations

| Protocol / window | Shared D std | Adam − SGD mean gap | Larger SGD − SGD mean gap | Head − SGD mean gap |
|---|---:|---:|---:|---:|
| initial / 0–511 | 0.185297–0.18634 | 0.147423–0.148674 | 1.50974–1.62403 | -0.0312984–-0.0291048 |
| initial / 0–299 | 0.182966–0.191512 | 0.10945–0.111023 | 1.61366–1.84306 | -0.0367138–-0.0339387 |
| initial / 300–310 | 0.169473–0.229261 | 0.162221–0.174071 | 0.93685–1.39787 | -0.0368619–-0.0255071 |
| intermediate / 0–511 | 0.185297–0.18634 | -0.0140547–-0.0139821 | -0.00689303–-0.00685449 | 0.00234173–0.00243739 |
| intermediate / 0–299 | 0.182966–0.191512 | -0.0111095–-0.0109581 | -0.00503968–-0.0047196 | 0.00150174–0.00166303 |
| intermediate / 300–310 | 0.169473–0.229261 | -0.0170333–-0.0166834 | -0.0105775–-0.00814706 | 0.0024173–0.00354869 |
| terminal / 0–511 | 0.185297–0.18634 | -0.000477208–-0.000473986 | -0.00109985–-0.00108002 | 7.92011e-05–7.96483e-05 |
| terminal / 0–299 | 0.182966–0.191512 | -0.000265644–-0.000260995 | -0.000713728–-0.000690004 | 4.4982e-05–4.7048e-05 |
| terminal / 300–310 | 0.169473–0.229261 | -0.000593603–-0.000569778 | -0.00150576–-0.00101568 | 7.23358e-05–0.000130419 |

The terminal protocol has roughly 10^-4–10^-3 average setting offsets; the initial protocol includes much larger responses. The 10^-3–10^-2 statement should therefore identify the intermediate protocol. The coarse display includes actual step 300 as well; its separate metrics are in JSON. Twenty-step means, if used, are display aggregation, not the registered 64-step statistics.

## These gaps are not constant

| Intermediate window 300–310: setting minus smaller SGD | Mean gap range | Across-step gap std range |
|---|---:|---:|
| momentum_low | 4.01795e-05–0.000402231 | 0.000698987–0.00122924 |
| adam_low | -0.0170333–-0.0166834 | 0.00125323–0.00451348 |
| sgd_high | -0.0105775–-0.00814706 | 0.00265954–0.00443677 |
| momentum_high | -0.00836253–-0.00763207 | 0.00292799–0.00347304 |
| head_low | 0.0024173–0.00354869 | 0.00173226–0.00233346 |

At the endpoint, **0/45 token-constant-gap tests pass in each protocol** under the registered std/abs(mean) ≤ 0.1 gate. At terminal smaller SGD vs Heavy Ball, token-centered RMS is 1.46029e−4–2.93560e−4 nats and std/abs(mean) is 46.49–163.48. Shared D cancels exactly in same-batch setting comparisons; it cannot establish strong RGI for the remaining response.

## Why the two larger rates dominate Q2

Q2 = (1/2) mean Var_p0(Δz). Therefore sqrt(2 mean Q2) measures a probability-weighted RMS centered logit displacement. If Δz is multiplied by c at fixed p0, Q2 is multiplied by c². Changing an optimizer rate does not hold the later gradient path or direction fixed.

| Terminal setting | Full-arrival Q2 high/low ratio | Corresponding RMS logit ratio | Zoom Q2 ratio | Zoom RMS logit ratio |
|---|---:|---:|---:|---:|
| sgd_low | 30.5247–35.3489 | 5.52491–5.9455 | 30.4266–38.4477 | 5.51603–6.20062 |
| momentum_low | 30.5784–35.7199 | 5.52977–5.97661 | 29.2165–38.9615 | 5.40523–6.24192 |

The terminal larger/smaller rate ratio is 10. At the first post-update arrival, measured Q2 ratios are 99.9715–100.0016 across the SGD and Heavy Ball pairs, agreeing with local quadratic scaling. Across accumulated training they become 30.52–35.72 in full-window means and 29.22–38.96 in the zoom. The realized high/low logit-displacement ratio is smaller than a simple 10× displacement prediction; this is a measured comparison, not a contraction theorem. A single linear y-axis consequently compresses the small-setting curves; use a labeled inset or the low-only supplement without artificial separation.

## Fixed probe and provenance

D is exactly shared across settings within each registered permutation, and exactly constant across the observed fixed-probe readouts. Cross-protocol D differences are at floating-point scale. The fixed probe is observed only at steps 0,16,32,64,128,192,256,320,384,448,512; there is no observed fixed-probe trace at 300–310. Full fixed-probe and arrival-window statistics, all 135 endpoint pair tests, high/low Q2 comparisons, and SHA-256 of every input are in figure-scale-audit.json.

Independent assertions passed: 20349. The independently recomputed full-window and final-token metrics agree with the frozen stats.json values.
