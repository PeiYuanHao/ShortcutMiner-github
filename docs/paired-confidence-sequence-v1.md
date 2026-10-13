# Opt-in paired screening protocol

The `paired-betting-cs-v1` confidence method is enabled only in the new
synthetic-corner-patch memory-v4 configuration. Configurations omitting this
field retain their exact-binomial intervals, look allocation, serialization,
and maximum probe fraction of 0.5. Completed artifacts are never rewritten.

For each independent protected-content group, D is baseline correctness minus
intervention correctness, taking values -1, 0 or 1. For proposed mean m, use
E_t(m) = sum_j w_j product_i (1 + lambda_j (D_i - m)).
Fixed stakes are [-.49, -.25, -.1, .1, .25, .49] and weights
[.2, .2, .1, .1, .2, .2]. All factors are positive on [-1,1]. At the true
constant conditional mean, each product is a nonnegative martingale starting
at one. Their normalized mixture is also one. Ville's inequality makes the
sublevel set E_t(m) < 1/alpha a simultaneous confidence sequence over all looks.
Each product is convex in m because all its affine factors have the same slope
sign, so the set is an interval; weighted log-domain computation and root
finding return its endpoints. Empirical mean belongs to the set by AM-GM.

Allocate alpha = .05/3 per candidate, with no second division by look count.
The union bound controls the budget decisions for up to three candidate probes
under the stated sampling assumptions. Delta remains .02; .01/.02/.05
sensitivity decisions are descriptive, not three separately tested discoveries.
Lower > delta expands model training; upper < delta stops; otherwise acquire
more pairs until exhaustion. The method does not guarantee narrower intervals
or a positive finding. No-op processing controls and task-learning checks remain.

Coverage assumes constant conditional mean for independent held-out groups,
conditional on training/validation-only model and construction selection. The
implementation estimates a superpopulation mean, not a fixed finite-dataset
mean under sampling without replacement. Repeated development on this dataset
does not become a new independent confirmatory study through a protocol change.

The new corner configuration reserves up to 1,500 of 2,000 ID groups for
checkpoints 128/256/512/1024/1500 and retains 500 for final independent paired
confirmation. Training remains conditional on the gate. No six-model matrix is
claimed when the gate defers. Original dataset OOD pilot diagnostics remain
separate and cannot overrule the paired gate.

The accompanying `candidate-bound-v2` Executive policy keeps only authorized
candidate-bound metadata on qualitative tasks. It adds no construction truth
and no metadata input to the RGB classifier. Legacy routing remains unchanged.

Reference: [Waudby-Smith and Ramdas, Estimating means of bounded random
variables by betting](https://arxiv.org/abs/2010.09686). The fixed finite mixture
above is an explicitly documented local implementation of the martingale
construction, not a claim to implement every adaptive or finite-population
variant in that paper.
