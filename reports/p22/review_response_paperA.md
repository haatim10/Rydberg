# Response to the external review — Paper A (HS-GS letter)

The review read a nine-page document. **The submission artefact in this
repository is the five-page letter** `paper/paper1/haatim_hsgs_letter.tex`,
which is a later and much-reduced version. Several of the review's P0 items
were already addressed there before the review was written, and I say so below
rather than claiming credit for fixing them. Four superseded drafts
(`paper/master/`, `paper/merged/`, `paper/spl1/`, `paper/summary/`) do still
carry the uncorrected wording; they are not submission targets and were left
alone.

---

## A8 — "exact projection": the review is right, and the letter was already careful

I reproduced the review's numerical check exactly. With $g=[1,2,3,4,5]$ and a
$3\times3$ Hankel matrix, one rank-1 step gives re-Hankelised singular values
$[9.6194, 0.1295, 0.0616]$ and a matrix of **rank 3**, not 1.

**One thing the review did not state, which is the cleanest disproof.** The map
is not idempotent: applying it twice moves the vector again, by
$\|T(T(g))-T(g)\| = 0.0987$. A projection satisfies $T\circ T = T$ by
definition, so no argument about the sets is needed — the operator fails the
definition directly.

The letter already called it "a Cadzow projection", already stated "one
application is *not* idempotent", and already used $n_{\mathrm{cz}}=4$ rather
than one pass. What the review is right about, and what has now been added:

- alternating projections between the rank set and the Hankel set **need not
  converge to the nearest point of their intersection**, so the operator is a
  denoiser and not a minimum-distance structured approximation;
- **low Hankel rank is a relaxation of the array manifold**, admitting
  exponential sums whose modes are not unit-modulus and therefore are not
  spatial frequencies of a ULA. Imposing it is not imposing the manifold.

## A7 — the split/refit protocol, and a review concern that does not apply

The review writes: *"If HS-GS uses only 70% of pilots and the comparator uses
100%, full rank does not give the same final estimate."*

**Checked in code, and it does not apply.** `rydberg_sim/track_b_proposed.py`,
`hs_gs_auto` calls `select_order_heldout(S, Z, ...)` to score candidates on a
held-out split, then calls `hs_gs(S, Z, ...)` with the **full** pilot matrix.
The split is used for selection only; the final estimate is refitted on all $P$
columns. HS-GS and EM-GS therefore consume the same pilot budget, and the
reported full-rank identity $\max|\text{diff}| = 0$ is exact rather than
approximate.

This was true of the implementation and absent from the paper. The letter now
states it.

## A3 — path-gain normalisation: correct in code, unstated in the paper

The review asks whether total received power changes with $L$. It does not.
`rydberg_sim/channel.py`: $\alpha_{\ell,k}\sim\mathcal{CN}(0,\beta_k/L_k)$, so
$\sum_\ell\mathbb{E}|\alpha_{\ell,k}|^2 = \beta_k$ independently of $L_k$. The
path-count sweep varies channel complexity at **fixed received power**.

Correct implementation, missing disclosure. Now stated where the channel model
is defined.

## A5 — the second generator is now named

The letter said only "a second, independently specified configuration". The
review is right that a generalisation claim cannot be evaluated without knowing
what it generalised to. It is now identified: a fixed $L=10$ paths per user,
$\alpha_\ell\sim\mathcal{CN}(0,1)$, angles uniform on $(-90^\circ,90^\circ)$,
contrasted with the clustered first generator, with only the receive-side
configuration borrowed.

## A10 — physical scope

Added, in the review's own terms: this is **one Rydberg architecture, not the
family.** A superheterodyne readout recovers amplitude *and* phase, and nothing
here applies to it — the problem is created by the magnitude-only readout, so
an architecture avoiding that readout avoids the problem.

---

## Already in the letter before the review

Recorded so the review's checklist can be closed honestly rather than re-done:

| review item | status in the letter |
|---|---|
| A2, "do not describe fallback as a guarantee against harm" | §V already says the statistic is "neither a guarantee about mean-square error" and reports both poolings side by side with the disagreement stated |
| A2, both summaries | both ratio-of-sums and paired median are reported for every aperture, with abstention fractions |
| A8, non-idempotence | already stated |
| A7, full-rank identity | already reported as $\max|\text{diff}| = 0$ |
| A5, scope of the cross-generator tests | already says these are cross-*generator*, not out-of-model, and that both stay within the ULA family |

---

## Not done this turn, and why

**A9 — derive or remove the bounds.** The review is right that
$U(U^\top J U)^{-1}U^\top$ is not a reproducible specification. This needs the
real parameter vector, the likelihood, derivative expressions, dimensions, a
tangent-space basis, rank conditions and the NMSE normalisation — a derivation,
not an edit. It is the largest single item outstanding on Paper A, and the
review's fallback ("if the CCRB cannot be verified, remove its curve and claims
rather than leaving an unauditable theory section") is the right instruction if
the derivation does not get written.

**A5, the trial counts.** The review asks why 276 and 266 trials appear when
the setup states 300–400. I did not find the answer in the result files this
turn. It needs checking before submission — most likely per-generator rejection
of degenerate draws, but I will not assert a cause I have not confirmed.

**A1/A4/A6 — the P1 evidence items.** Paired uncertainty on the headline gain,
a held-out comparison of $\rho$ against simpler descriptors such as $L$ and
$L/r_{\max}$, and per-condition one-shot-versus-repeated curves. These are the
items that decide whether Paper A's effective-rank contribution survives, and
the review is right that they are the difference between a characterisation and
a claim. None is a wording change.

**A10 formatting items** — Eq. (9)'s missing left-hand variable, the $q/y$
inconsistency, undefined Table I labels, Fig. 2(b) line styles. These are
defects of the nine-page document the review read; the five-page letter has a
different equation numbering and table, and I did not find these specific
defects in it. They should be re-checked against whichever version is
submitted.

**The author list of `precoding2408`** is taken from `paper/master/refs.bib`
and is **not verifiable from this container**; `reports/trackD_step0_cui_prediction.json`
records `citation_verified: false`. The title and arXiv identifier were
confirmed by the author. Added to `docs/open-todos.md`.
