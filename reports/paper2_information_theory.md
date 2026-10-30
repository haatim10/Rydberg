# What the information filter guarantees — and what it does not

Consider one correction in normalized real coordinates. Let J be symmetric positive semidefinite, tau >= 0, and d any proposed structural correction. Define delta = (I + tau J)^(-1)d.

If J = Q diag(mu_i) Qᵀ with mu_i >= 0, then the coefficient in direction q_i is

delta_i = d_i / (1 + tau mu_i).

Thus ||delta||₂ <= ||d||₂. If J >= m I, then ||delta||₂ <= ||d||₂/(1+tau m). The correction is the unique minimizer of

0.5 ||delta-d||₂² + 0.5 tau deltaᵀJ delta.

This is a local quadratic penalty on measurement-sensitive directions, not an exact likelihood constraint. For tau>0,

deltaᵀ J delta <= ||d||₂²/(4 tau),

because mu/(1+tau mu)² <= 1/(4 tau). This bound alone says nothing about channel-estimation error.

For the magnitude model with complex noise variance v, amplitude information is I_a(v,a)=q(a/sqrt(v))/v. Here q(0)=0 and q(t) tends to 2 as t grows. For a fixed nonzero field, I_a therefore behaves like 2/v as v approaches zero.

If the measurement amplitude derivatives span all real channel coordinates, their weighted Gram matrix stays uniformly positive definite, the estimated fields remain away from zero, tau is bounded below by a positive number and ||d|| stays bounded, then the smallest eigenvalue of J grows on the order of 1/v and ||delta|| tends to zero on the order of v. These are substantial assumptions. They may fail for poor estimates, repeated/insufficient pilots, vanishing fields, or an ambiguous reference geometry.

If J has a nullspace, the filter leaves the component of d in that nullspace unchanged. More generally, for J(v)=A/v with fixed positive semidefinite A, delta tends to the projection of d onto ker(A), provided d has a limit. This is why a low-noise claim without a rank condition would be wrong.

The per-antenna measurement blocks can be assembled into a block-diagonal full information matrix: the observations are independent across antennas conditional on the channel. The Hankel correction can still couple antennas; the block-diagonal filtering step solves the stated quadratic problem for that already formed full correction.

The guarantees apply to one correction calculated from J. They do not establish global identifiability, reduction of actual likelihood, reduction of channel error, physical correctness of a rank model, or convergence of the alternating estimator. Estimated J can be wrong in precisely the directions where the current channel estimate is wrong. The numerical experiments must determine whether this local property is useful.
