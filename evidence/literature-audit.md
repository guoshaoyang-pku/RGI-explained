# Literature audit: A Journey to the Edge of Stability

Read on 2026-10-08. Source: Jaerin Lee and Kyoung Mu Lee, arXiv:2609.32290v1 (2026-09-26). This note distinguishes verified source claims from a proposed bridge to the RGI decomposition.

## Verified scope

- The task is CIFAR-10, fixed batch size 4096, with a two-layer tanh MLP, a four-layer tanh CNN, and a small three-block ViT (the classification head is excluded from those layer counts).
- Ten methods are fixed causal linear filters of the gradient: SGD; Heavy Ball and Nesterov with beta 0.5/0.9; two QHM variants; and three two-pole Grokfast variants.
- Runs use the same model initialization. The authors sweep raw learning rates and run each case for 200k iterations.
- Preconditioning is explicitly out of scope (PDF p.2). Thus the empirical result does not cover Adam, RMSProp, Muon, or architecture interventions.
- The effective step is h = eta Q(1), with Q(1) the dc gain of the fixed gradient filter. Normalized Heavy Ball has Q(1)=1; unnormalized Heavy Ball has Q(1)=1/(1-beta).
- Figure 1 compares final/stabilized scalar measurements across effective learning rates. Figure 3 compares loss and sharpness over progress tau = t eta Q(1), showing nearly coincident low-LR traces and different high-LR traces.
- The reported gradient cosine is consecutive-gradient alignment inside one run, c_t = cos(g_t,g_{t+1}). It is not cross-optimizer parameter, update, or gradient alignment.
- The low-LR numerical boundary 5e-5 in Table 2 belongs to their MLP/CIFAR setting. It must not be transplanted as an LLM threshold.

## Local mathematical explanation and its assumptions

Appendix B (PDF pp.15-19) declares local small-h expansions for a fixed smooth objective and a fixed stable filter at an anchored iteration. Its quasi-dc step assumes gradients vary slowly through the optimizer memory. In that regime:

\[
\theta_{t+1}=\theta_t-hg_t+h^2\frac{Q'(1)}{Q(1)}H_tg_t+O(h^3),
\qquad
\frac{d\theta}{d\tau}=-g-h\alpha_QHg+O(h^2),
\quad \alpha_Q=\frac12-\frac{Q'(1)}{Q(1)}.
\]

The optimizer's normalized leading response is shared; filter shape appears in the next correction. This is a useful motivation for a dc-matched SGD/momentum comparison. It is not a proof for state-dependent preconditioners or independently changing stochastic batch objectives.

The paper's first-order alias is rho_t = |alpha_Q| ||g_{t+1}-g_t|| / ||g_t|| = O(h), whereas 1-c_t = O(h^2). A gradient cosine very near one can therefore miss a first-order deviation. Prefer an actual paired function-response comparison in our experiment.

## Connection to our four terms: a proposed, testable bridge

For branch m from a common theta_0 and a common measurement batch B:

\[
L_m(B)=D(B)+A_m(B)+Q_{2,m}(B)+\varepsilon_m(B),
\quad A_m=g_B(\theta_0)^\top\delta\theta_m,
\quad Q_{2,m}=\tfrac12\operatorname{mean}\operatorname{Var}_{p_0}(\Delta z_m).
\]

D(B) is common by construction. It cancels exactly in a paired method difference:

\[
L_a-L_b=(A_a-A_b)+(Q_{2,a}-Q_{2,b})+(\varepsilon_a-\varepsilon_b).
\]

The two statements to test separately are: (i) frozen batch difficulty explains arriving-batch loss fluctuation; and (ii) the remaining response terms explain cross-method differences. The first alone does not prove RGI or common parameter dynamics.

If parameter displacements genuinely align after a justified time normalization, then |A_a-A_b| <= ||g_B(theta_0)|| ||delta_theta_a-delta_theta_b||. Let C_B be the frozen categorical covariance seminorm over batch logits. The variance-term difference obeys

\[
|Q_{2,a}-Q_{2,b}|\le \tfrac12\|\Delta z_a-\Delta z_b\|_{C_B}
  (\|\Delta z_a\|_{C_B}+\|\Delta z_b\|_{C_B}).
\]

Thus verified trajectory/function alignment would constrain A and Q2, while the exact measured remainder still needs checking. Overlapping mean loss or sharpness alone does not establish these bounds are small.

A useful local scale warning: A is first order in parameter displacement; Q2 is second order in actual logit displacement. Epsilon contains both network nonlinearity R = W-A and the log-sum-exp approximation Q-Q2. R is generally second order in parameter displacement; epsilon is not generically a cubic remainder, and cannot be assumed smaller than Q2.

## What the common late-checkpoint experiment should measure

1. Use exactly the same checkpoint, tokenizer, parameters, prefixes, targets, masks, and arriving-batch order. Warmup/restarts are branch interventions with documented optimizer-state reset.
2. Use a fixed probe in addition to arriving batches: D is constant over time on the fixed probe, but can oscillate on the prescribed arriving-batch stream. Do not conflate these two plots.
3. Keep the requested four-panel plot on a common training-step/token axis and show warmup explicitly. A, Q2 and epsilon must all be zero at the shared checkpoint, up to measured numerical tolerance.
4. Add an optional predeclared progress axis tau_t = sum_s eta_s Q(1) for fixed linear-filter methods. Do not fit a post-hoc time warp from loss or four-term curves. Do not assign Adam a scalar dc gain.
5. A dc-matched SGD/normalized-HB pair tests the published low-LR mechanism directly. Adam and interventions beyond optimizers are separate tests, with their calibration disclosed.
6. Same checkpoint does not ensure the same function if an intervention changes forward computation (e.g. changed architecture). Either make it function-preserving at t0, or measure the intervention's initial offset rather than claiming a zero initial response.
7. Inspect paired differences and each term's contribution, not only raw-loss correlation. Cross-method loss offsets must be tested across tokens/prefixes as well as batch averages before claiming strong RGI.

## Optimizer restart is a real finite-time confound

With normalized Heavy Ball m_t=beta m_{t-1}+(1-beta)g_t, zero m_{-1}, and a constant initial gradient, m_t=(1-beta^(t+1))g_0. Its first updates are smaller than SGD even when their dc gains match. Under a prescribed schedule h_t the cumulative displacement is -sum_t h_t(1-beta^(t+1))g_0, versus -sum_t h_t g_0 for SGD. This discrepancy is a startup transient, not a falsification of the established-filter quasi-dc approximation.

Warmup should remain visible in the main plots. A separate test can start after the transient or measure an analytical startup correction, but should not conceal it with empirical time alignment. Small learning rate also does not eliminate stochastic batch noise: same batches control a nuisance, while large batch/fixed-objective probing is needed to diagnose the quasi-dc assumption.

## Safe citation language for the paper

> Lee and Lee (2026) observed nearly coincident low-learning-rate loss and sharpness traces after dc-gain normalization for a family of linear gradient filters on CIFAR-10. Their local modified-flow expansion places filter-specific corrections beyond the shared leading gradient-flow term. We test a complementary, sample-conditioned question: whether frozen difficulty and measured finite-response terms account for the loss structure and differences of language-model continuations from a common checkpoint.

Avoid: "all optimizers follow the same parameter trajectory," "Adam is covered by dc matching," or "the edge-of-stability paper already explains token-wise RGI."

## Evidence and source locations

- arXiv abstract: https://arxiv.org/abs/2609.32290v1
- PDF: https://arxiv.org/pdf/2609.32290v1
- HTML: https://arxiv.org/html/2609.32290v1
- Verified BibTeX: https://arxiv.org/bibtex/2609.32290
- Independent metadata: https://api.datacite.org/dois/10.48550/arxiv.2609.32290 (title, both authors, year 2026, publisher arXiv match).
- PDF SHA256: 2d7d9f3faab43ae23d1ae06568d5e78c26557b55e9fd39a08ff16fb3cfd1496a
- Abstract HTML SHA256: 71d7fbbc4de28f8fdd56560947a14ad32513e53eb2a6daba84105bd9f328007b
- Source HTML SHA256: 47cc97e8ef20951004715d1ec22a8972b541384a4685c9dd8113795632861af1
- Most relevant verified passages: PDF p.2 preconditioning exclusion; p.3 experiment/observables; pp.6-7 low-LR overlap; p.17 Theorems B.1/B.2; pp.18-19 alias/alignment; p.20 optimizer/regime specifications.

## Retrieved BibTeX

```bibtex
@misc{lee2026journeyedgestability,
      title={A Journey to the Edge of Stability},
      author={Jaerin Lee and Kyoung Mu Lee},
      year={2026},
      eprint={2609.32290},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.32290},
}
```

## Alignment is not equivalent to token-wise RGI

At matched progress, genuinely identical parameters give identical conditional losses and a zero gap. At different progress, sharing a leading gradient-flow direction only gives the local sample response -delta_tau * g_i(theta_0)^T g_train(theta_0). This quantity generally depends on the token/prefix i. An approximately constant nonzero loss offset requires the sample-centered paired response (Delta A + Delta Q2 + Delta epsilon) to be small, rather than merely the leading parameter direction being shared. Thus dc-normalized flow universality motivates controlled comparisons, but does not derive the stronger token-wise RGI claim.

Also, two variance ratios with different denominators answer different questions: SSE(L-D)/SST(L) assesses frozen prediction of arriving-batch losses; SSE((L-D)-A-Q2)/SSE(L-D) assesses response closure. For method differences, use SSE(Delta L-Delta A-Delta Q2)/SSE(Delta L), excluding exact t0 and reporting absolute nats when the denominator is numerically small. A, Q2 and epsilon may covary; their individual variance percentages are not additive causal shares.

## A precise Hessian bridge and a separating control

For one scored prefix/target, write r=p0-e_y, J0=D_theta z(theta0), C0=diag(p0)-p0 p0^T, and H_v=D_theta^2 z_v(theta0). The parameter loss Hessian splits exactly at the checkpoint:

\[
\nabla_\theta^2\ell(\theta_0)=J_0^\top C_0J_0+\sum_v r_vH_v.
\]

For a small displacement delta, averaged over scored tokens,

\[
A=g_B^\top\delta,\quad
Q_2=\tfrac12\delta^\top\langle J_0^\top C_0J_0\rangle_B\delta+O(\|\delta\|^3),\quad
R=\tfrac12\delta^\top\left\langle\sum_v r_vH_v\right\rangle_B\delta+O(\|\delta\|^3).
\]

The exact observed-logit Q2 is nonnegative, but is not equal to the full parameter-Hessian response. This connects the four-term measurement to the curvature in the edge-of-stability paper: its sharpness is the maximum eigenvalue of the full Hessian, which includes both the generalized Gauss-Newton/Fisher term and the residual network Hessian. One cannot identify scalar sharpness S with our directional observed-logit variance Q2.

The registered head-only branch supplies a clean separating control. Since Pythia has a separate output matrix and all preceding hidden states remain frozen, logits are affine in the only updated matrix. Hence Delta z=J0 delta exactly, W=A exactly, and R=0 up to numerical precision. The measured epsilon then equals Q-Q2 alone. For a small head update, this softmax remainder begins with the third centered cumulant of Delta z divided by six. Compare its measured size with full-network R to identify the actual approximation obstruction, rather than treating epsilon as an unspecified catch-all.
