# Theory and scope review

## Equation checks

- The natural fixed-reference decomposition uses a measured checkpoint reference. It is not labelled a natural Bayes distribution or a Bayes excess risk.
- For an observed displacement, L-D=W+Q is an exact log-partition identity. The manuscript explicitly calls it an arithmetic identity and does not call it a future-update predictor.
- A is the frozen-start batch gradient dotted with the observed parameter displacement. Q2 is the second-order logit cumulant computed from the observed logit displacement. A+Q2 is therefore a retrospective finite-response approximation.
- R=W-A is kept as a representation/nonlinearity residual; it is not silently identified with a complete Hessian or optimizer term.
- The known-teacher decomposition ell=S+C+E correctly gives E_q[C|x]=0 only under the fixed teacher population. The natural sampled-target case is not given that simplification.

## Dominance and failure checks

- Raw loss dominance is attributed to frozen difficulty using a common official step-32000 reference and separately using each algorithm's frozen start.
- Slow-response dominance is reported with signed covariance shares. Shares above one and negative shares are explained as cancellation-sensitive attribution coefficients.
- The scalar probe failure, SGD high-pass failure, and optimizer-difference failure remain in the paper. They prevent a claim that one global slow variable or two terms close every frequency band and paired optimizer gap.
- The known-teacher constant-offset failure is reported after exact population integration. The support-gate construction is marked sufficient only and is not presented as a natural-LLM explanation.

## Evidence limits

The three permutations reuse one data pool and are not independent datasets. The local continuation is Pythia-70M and does not reproduce the original RGI paper's 124M/0.7B from-scratch training. Independent verifiers audit saved traces, hashes, identities, aggregations, and fixtures, but do not rerun full network gradients or complete optimizer trajectories. These limits are stated in the manuscript and evidence registry.
