# Common late-checkpoint RGI continuation

Frozen before science measurements, 2026-10-08. Owner: root/rgi_common_ckpt_20261008.

## Inputs and interventions

- Official EleutherAI/Pythia-70M step143000, revision de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852. Local converted checkpoint SHA256 6b81e9cca49e921014e68e3bdc1400a2c3e69e46b46bace57814ce110ee17467. This is late Pile pretraining, not a checkpoint equilibrated on FineWeb.
- Reuse registered FineWeb sample10BT extension pool: 4096 train /64 calibration /512 held-out documents; one256-token span/document. Score223 positions33..255, vocabulary50304. Reusing this pool is a controlled follow-up, not a fresh-test claim.
- Same initial parameters and forward function in every branch. Optimizer states reset. No dropout. FP64 parameters/gradients/logits/reductions; TF32 disabled.
- Three paired permutations, seeds202610081, 202610082, 202610083. These share one data pool, not three independent datasets.512 updates, batch8, no replacement within each permutation. Measurements on arriving batch B_t precede its update; t0 is the common checkpoint. Linear warmup32 updates to a constant rate thereafter.
- Six settings: SGD low effective rate h=0.0003; normalized Heavy Ball beta0.9 at the same h; reset Adam beta(0.9,0.95),eps1e-8 with a calibrated rate; SGD high h=0.0012; normalized Heavy Ball high at the same h; output-head-only SGD low at h=0.0003. Head-only is a parameter-update mask, with unchanged initial forward computation. No weight decay.
- Heavy Ball implementation uses PyTorch SGD momentum0.9 and raw lr=(1-beta)h. Its dc gain is1/(1-beta). Zero initial momentum gives a real startup transient, retained in plots. Warmup and common stochastic batches do not ensure quasi-dc gradients.
- Adam's single scalar rate is calibrated BEFORE training on first16 disjoint calibration documents, using a reset one-step full-gradient update, to match SGD-low full-vocabulary Q2. Start at1e-6; rescale by square-root ratio, at most two corrections. No final loss, evaluation data, or curve selection enters calibration. This is an initial functional-step scale match, not a dc-gain equivalence or a guarantee of matched training progress.

## Measured terms and records

For all scored positions on the same B_t, record L,D,W,Q,Q2,A,R,epsilon:

    D=CE(theta0,B_t)
    W=mean[(p0-e_y) dot (zt-z0)]
    Q=mean[KL(p0||pt)]
    Q2=.5 mean[Var_p0(zt-z0)]
    A=gradient_theta CE(theta0,B_t) dot (theta_t-theta0)
    R=W-A
    epsilon=R+(Q-Q2)=L-D-A-Q2

A is measured using actual displacement; this explains a realized response rather than forecasting unknown parameter motion. Epsilon contains generally second-order network nonlinearity, not only a third-order softmax remainder.

Store all batch and document means, rate schedule, input/code hashes, current displacement norm and exact full-vocabulary8-token fixtures. At t0 verify A=Q2=epsilon=0. Verify W+Q identity tokenwise to2e-10. Fixed held-out32-document probe at steps0,16,32,64,128,192,256,320,384,448,512: frozen D is constant, saved gradient supplies A. At final step, save tokenwise A via a full-model JVP and compare its mean to the independent parameter-gradient dot product. Save final parameters and fixed-probe logits only when needed for an independent numerical audit; weights/logits are local artifacts, not automatic public-release content.

## Prospective questions and gates

1. Frozen difficulty dominance on arriving batches: report std(D),std(L),RMS(L-D), and 1-SSE(L-D)/SST(L). Pass only if the last quantity>=0.90. A shared coefficient1 is an identity, not empirical evidence of matched dynamics.
2. Finite-response closure: RMS(epsilon)/RMS(L-D)<=0.10; separately report absolute nats, raw, highpass64 and blockmean64, full0:512, near0:128, far384:512. Equality after adding epsilon is arithmetic, not a passed approximation gate.
3. Paired method differences: D must be equal; compare DeltaL with DeltaA+DeltaQ2, using the same10% relative RMS gate. Report centered fixed-probe token gaps, std/abs(mean) and centered RMS on low-variance denominators. Strong constant-gap gate is0.10, but near-zero means are flagged as undefined and evaluated in absolute nats; a zero gap is distinct from nonzero constant-offset RGI.
4. Alignment: SGD versus dc-matched Heavy Ball only, original shared step/token axis and prespecified integrated effective schedule, with warmup visible. Report actual parameter-displacement cosine and relative distance at512, fixed-probe loss-response difference and low/high contrast. No fitted time warp and no universal Adam alignment claim.
5. Term attribution: report signed covariance shares Cov(term,M)/Var(M),M=L-D; sum including epsilon is1. These are descriptive, can be negative, and are not independent causal percentages. Also report drop-one approximation errors and R versus Q-Q2 to identify the obstruction.

## Bounded execution

Run a64-update engineering pilot before the registered512-update jobs, on explicitly idle assigned A100 GPUs after current preflight. Pilot observations determine runtime/precision viability only, not checkpoint/data/method choice. Budget900seconds/job, stop on nonfinite values/identity failure/OOM. Do not modify or kill other workloads. Register launch command,GPU UUID,PID,source hashes and input hashes; harvest and run an independent checker before manuscript claims.
