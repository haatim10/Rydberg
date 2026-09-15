# PROMPT 22 Part D — the cheap measurements

Scripts committed before they ran. Nothing here changes an estimator; D1 is the
committed B5 cell re-run with two extra quantities recorded, D2 and D3 are
reporting repairs against numbers already in the repository.

---

## D1 — the placement comparison

`scratch/p22_partD_placement.py`, `results/p22/p22_d1_placement.json`.
300 paired trials per SNR, configuration A, same worlds and seed base as
`results/p12/B5.json`.

**Reproduction check first.** The re-run returns the committed cell's numbers
to the digit: $-0.0812$, $-0.1637$, $+0.0336$, $+0.0100$, $+0.0538$, $+0.1684$,
six-point mean $+0.0035$. So the new quantity is measured on the same
measurement, not a different one.

| SNR (dB) | interleaved $-$ post-hoc | 95% CI | excl. 0 | median $q$ | $q$ p90 | $q$ max |
|---|---|---|---|---|---|---|
| $-5$ | $-0.081$ | $[-0.159,+0.047]$ | no | **0.546** | 0.781 | 1.121 |
| $0$ | $-0.164$ | $[-0.317,-0.066]$ | **yes** | 0.266 | 0.374 | 0.665 |
| $5$ | $+0.034$ | $[-0.050,+0.133]$ | no | 0.134 | 0.198 | 0.335 |
| $10$ | $+0.010$ | $[-0.071,+0.077]$ | no | 0.071 | 0.102 | 0.225 |
| $15$ | $+0.054$ | $[-0.014,+0.107]$ | no | 0.040 | 0.056 | 0.123 |
| $20$ | $+0.168$ | $[+0.102,+0.246]$ | **yes** | 0.022 | 0.030 | 0.042 |

Paired intervals on the headline number, which it did not have:

- six-point mean $+0.0035$ dB, 95% CI over the six points $[-0.117,+0.124]$;
- pooled per-trial median $+0.018$ dB, 95% CI $[-0.008,+0.049]$.

Both contain zero. **"0.003 dB" was never a precise zero; it is an unresolved
average.**

### What the divergence shows

$q_i=\|\hat\bG^{\text{inter}}_i-\hat\bG^{\text{post}}_i\|_F/\|\bG_i\|_F$.

At $-5$ dB the two estimates differ by **55% of the channel norm** while their
errors differ by 0.08 dB. At $0$ dB, 27% against 0.16 dB. Even at $20$ dB,
where they agree most closely, the gap is 2.2%.

**The review's objection is confirmed as measured fact.** A difference of means
cannot establish that two estimators agree, and here it does not: they produce
visibly different estimates that happen to be about equally accurate. The gap
shrinks monotonically with SNR, which is the sensible direction — as the data
gets better both estimators are pulled towards the same answer.

### The sign change

Interleaving is significantly worse at $0$ dB and significantly better at
$20$ dB, by similar magnitudes. The near-zero mean is a cancellation, not an
absence of effect. The manuscript previously reported only the mean.

### The registered prediction failed

P18 predicted interleaving would be worth $[0.3,1.5]$ dB pooled. Measured
$+0.003$. This is one of the two registered failures now disclosed in the
manuscript.

### What the manuscript now says

§V-G is rewritten. The conclusion is narrower than before and, on this
evidence, more useful: the two placements are equally accurate on average while
being visibly different estimators; because post-hoc is far cheaper (one Cadzow
step instead of $T$), **it is the better practical choice here, and the paper
says so even though it is not the variant it proposes.** What is not claimed is
interchangeability.

---

## D2 — intervals on the negative results

Both existed in the repository; the manuscript quoted the point estimates bare.

**Path-count sweep** (`trackB_hankel_emgs/results/experiment_C_path_count.csv`,
300 trials/row). The paper reported only the endpoints, $7.04$ and $-0.12$ dB.
It now reports the whole sweep with the distinction the endpoints hid:

| $L$ | gain (dB) | 95% CI |
|---|---|---|
| 2 | $+7.043$ | $[+6.735,+7.335]$ |
| 4 | $+3.556$ | $[+3.379,+3.729]$ |
| 6 | $+1.792$ | $[+1.636,+1.956]$ |
| 8 | $+1.038$ | $[+0.927,+1.155]$ |
| 10 | $+0.577$ | $[+0.455,+0.695]$ |
| 12 | $+0.266$ | $[+0.191,+0.339]$ |
| 14 | $+0.046$ | $[-0.050,+0.136]$ — **contains zero** |
| 16 | $-0.117$ | $[-0.206,-0.038]$ — **excludes zero** |

At $L=14$ the constraint has stopped helping. At $L=16$ it is doing measurable
harm. Only the second says the constraint should be switched off, and the
manuscript now makes that distinction explicitly.

**Forced-active diagnostic** (`results/p12/B2_forced.json`, row `N16_L14`).
At $\rho=0.735$: $-0.075$ dB, 95% CI $[-0.186,-0.026]$, excluding zero. The
abstention rate at that point is $38.8\%$ — now stated, because it is what the
unforced estimator is protected by, and the file's own header warns that the
forced column must never be quoted as HS-GS.

---

## D3 — setup gaps

§V-A now defines: normalized error and the per-trial dB gain as equations; the
three summaries (paired median, mean dB gain, ratio of summed errors) and why
they are not interchangeable; how the twelve operating points are aggregated
(unweighted mean of per-point summaries, not a pooled statistic, with the
reason); $T$, the order-search budget and $\ncz$ in Table I; both generators;
and the six SNR values.

**The 276/266 trial counts are stated as measured with the cause explicitly not
asserted.** I did not find the answer in the result files. The likely
explanation is per-generator rejection of degenerate draws, and that remains a
guess, so it is in `docs/open-todos.md` as a must-resolve-before-submission
item rather than written into the paper as an explanation.
