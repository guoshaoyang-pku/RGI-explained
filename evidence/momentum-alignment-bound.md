# A finite-time bound for the registered normalized-momentum pair

This is an auxiliary analytical result, derived here rather than attributed to Lee and Lee. It separates identical-data coupling, optimizer memory, and sample-conditioned loss responses.

## Exact update algebra

Let both methods start at theta0 and use the same sequence of batch objectives f_t. Define SGD s_(t+1)=s_t-h_t g_t(s_t), and normalized Heavy Ball m_(t+1)=m_t-h_t v_t, where v_t=beta v_(t-1)+(1-beta)g_t(m_t), v_(-1)=0. The actual registered PyTorch SGD implements this v with raw_lr=(1-beta)h_t. Put e_t=m_t-s_t and k=beta/(1-beta).

For every T, without a quasi-dc or constant-gradient approximation,

\[
e_T=-\sum_{t=0}^{T-1}h_t\{g_t(m_t)-g_t(s_t)\}
 +k\left[h_{T-1}v_{T-1}-\sum_{t=0}^{T-2}(h_{t+1}-h_t)v_t\right].
\]

Proof: the momentum recurrence gives v_t-g_t(m_t)=k(v_(t-1)-v_t); substitute in the paired cumulative update and use summation by parts. Thus the two sources of the paired displacement are feedback from divergent parameters, and an explicit memory/schedule defect.

## Bound with stated regularity assumptions

Suppose every batch gradient is L-Lipschitz on a region containing both trajectories, ||g_t(m_t)||<=G, beta in [0,1), and all h_t>=0. Then ||v_t||<=G and

\[
\|e_T\|\le\sum_{t<T}h_tL\|e_t\|+
 kG\left(h_{T-1}+\sum_{t<T-1}|h_{t+1}-h_t|\right).
\]

For a nondecreasing warmup followed by a constant rate h_max, the last factor is at most 2h_max. Discrete Gronwall therefore yields

\[
\|e_T\|\le 2\frac{\beta}{1-\beta}h_{\max}G
\exp\left(L\sum_{t<T}h_t\right).
\]

For a constant rate h, the schedule defect simplifies to k h v_(T-1); its norm is <=k h G and the leading factor 2 can be replaced by 1. This is a finite-h bound for same-batch coupled SGD and momentum; stochastic objectives need not be individually quasi-dc. It does assume uniform smoothness and bounded gradients in a common trajectory neighborhood. The bound may be very loose over long horizons, especially with large L or beta near 1.

The low-h alignment limit is at fixed integrated progress sum h_t. At a fixed number of updates, lowering h can make all responses tiny and generate trivial raw-loss agreement; report their relative paired response and absolute nats separately. The registered h=.0003/.0012 pair tests the error's rate sensitivity but does not empirically verify these regularity constants.

## Consequences for the four terms

On a fixed probe B, its frozen gradient g_B(theta0) is shared. Hence

\[
|A_m(B)-A_s(B)|\le\|g_B(\theta_0)\|\,\|e_T\|.
\]

Let C_B denote the frozen categorical covariance seminorm on probe logits. With a K_z-Lipschitz logit map in this seminorm,

\[
|Q_{2,m}-Q_{2,s}|\le\tfrac12K_z\|e_T\|
 (\|\Delta z_m\|_{C_B}+\|\Delta z_s\|_{C_B}).
\]

The exact paired gap still includes Delta epsilon, which must be measured. Bounds on A/Q2 do not force a token-independent nonzero offset; its centered counterpart requires sample-conditioned readouts.

## Implementation cautions

- These recurrence equations require the specified PyTorch SGD semantics: dampening0, NesterovFalse, weight_decay0. Different dampening or beta schedules change the algebra.
- h_t is effective rate; raw PyTorch momentum rate is (1-beta)h_t. The registered zero momentum state is covered exactly.
- Warmup is part of the dynamics and appears explicitly in the summation-by-parts term; it need not be removed or fitted away.
- Adam and head-only comparisons are not covered by this pair bound. Their directional dynamics differ by construction.
- This is an auxiliary explanatory inequality, not a numerical validation that LLM SGD and momentum actually align. That requires observed paired displacement/function-response evidence.
