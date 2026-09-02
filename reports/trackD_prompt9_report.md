# PROMPT 9 — overnight run: normalization, classical scaling, matched training

Claims tagged **[FACT]** (measured or verified, with `file:line`), **[MATH]**
(derived), **[HYP]** (needs the named experiment).

Primary statistic everywhere: the **paired per-trial median, per SNR bin**.
Pooled scalars appear only where a whole-cell number is needed for a table, and
carry the label `pooled_SAMPLING_DESIGN_DEPENDENT` in the JSON — the SNR draw
is uniform by construction, so a pooled median reports the sampling design as
much as the estimator.

Pre-registration: `reports/trackD_prompt9_prereg.md`, committed standing alone
as `e24c62a` **before** any Part B sweep or Part C training run started.

---

## Headline

Four things came out of the night, in order of how much they change the story.

1. **The SNR-balanced loss is the single largest free win in the project so
   far.** +1.90 dB at SNR ≥ 5 for **no cost at all** at low SNR (+0.14 dB, also
   positive). It closes the high-SNR gap to the unstructured-LS oracle from
   +2.90 dB to +1.00 dB. This was pre-registered as a trade-off; there was no
   trade-off.
2. **Pilot efficiency and pilot-count generalization are different quantities**,
   and the paper's Fig. 4 does not say which it plots. Matched training at
   `P = 10` beats the `P = 20` model evaluated there by **+2.23 dB**.
3. **Both Part A normalizations collapsed.** `K` cancels out of the compression
   ratio [MATH]; re-indexing by effective rank moves the zero crossing to
   `r_eff/cap = 0.518` and turns a curve about one simulator's `L` into a
   statement about any channel.
4. **The structural prior's payoff is robustness, not accuracy.** G1 is behind
   URformer in distribution but degrades least out of it (+0.48 dB vs +0.73 dB
   under a path-richness shift, against a flat oracle).

---

## 1. Part A — normalization re-analysis (no compute)

Full detail in `reports/trackD_normalization.md`; the load-bearing results:

### A1 — `K` cancels out of the compression ratio [MATH]

```
3 Σ_k L_k / (2NK)  →  (L_k ≡ L̄)  →  3 K L̄ / (2 N K)  =  3 L̄ / (2 N)
```

Verified numerically for all `(K, L̄) ∈ {2,3,4,6} × {3,5,7}` to machine
precision. At the default (`L̄ = 5, N = 32`) the structural model uses **23.4%**
of the unstructured degrees of freedom, independent of the number of users.

### A2 — the `r_eff` collapse [FACT]

Re-indexing Track B's Experiment C by the median Roy–Vetterli effective rank of
the noiseless channel columns moves the zero crossing from `L/cap = 0.90` to
**`r_eff/cap = 0.518`**. `L` is a property of one generator; `r_eff` is a
property of any channel, so the useful region becomes **`r_eff/cap ≲ 0.5`** — a
claim that can be checked on somebody else's channel model, which is what B6
does.

### A3 — the EM filter's value collapses onto `κ = 2Z|Y|/σ²` [FACT]

**`Δ_{GS−EM-GS} · κ = 5.32`** (median; range 4.51–6.17) while `κ` itself spans
**375×**. A second family — `P` swept at fixed SNR, where `κ` is constant by
construction — confirms it. So the finding that survives is not "the Bessel
filter is inert in our regime" but

> **`Δ_{GS−EM-GS} ≈ 5.3 / κ` dB** — the EM filter earns its place only when
> `κ ≲ 5`.

[HYP] that 5.3 is universal rather than RSR-specific; settling it needs a second
RSR, which this prompt did not authorize.

### A4 — data scaling [FACT]

At 80k the model sees **0.05 samples per parameter** (64.5 real measurements per
parameter), which is why the 20k→80k gain of 1.35 dB had not saturated.

### A5 — what the paper does and does not state [FACT]

Xiao et al. describe Fig. 4 only as "evaluated at a fixed SNR of 5 dB". **The
paper does not state whether the networks were retrained per pilot count.** Part
C resolves the question for our own curve — see §3.2 and the three-way figure.

### One correction to the brief

The brief cites a "0.068 dB GS-vs-EM-GS gap". **I cannot source that figure in
this repository.** My nearest measurement is **+0.060 dB at `κ = 76.7`
(SNR = +10 dB)**. I report what I can trace.

---

## 2. Part B — classical scaling sweeps (no training)

<!-- FILLED IN WHEN THE CELLS LAND -->

---

## 3. Part C — matched training

Pre-registration in full: `reports/trackD_prompt9_prereg.md`. Four runs, each
80k samples × 13 epochs, epoch selected by the stage-2 rule on held-out
validation:

| run | condition | best val | epoch |
|---|---|---|---|
| C1 | SNR-balanced loss, `P = 20` | −6.768 dB | 8 |
| C2 | URformer, matched `P = 10` | −4.611 dB | 7 |
| C3 | URformer, matched `P = 35` | −7.938 dB | 6 |
| C4 | G1 gated Hankel, matched `P = 10` | −4.499 dB | 8 |

### 3.1 The balancing did what it was designed to do [FACT]

Realized per-bin **gradient** shares, measured (not assumed):

| bin | −10..−5 | −5..0 | 0..5 | 5..10 | 10..15 | 15..20 |
|---|---|---|---|---|---|---|
| static weight | 0.056 | 0.111 | 0.310 | 0.687 | 1.641 | 3.195 |
| share, uniform loss | 0.465 | 0.293 | 0.138 | 0.061 | 0.027 | 0.015 |
| share, balanced | 0.109 | 0.137 | 0.181 | 0.176 | 0.188 | 0.209 |

Spread across bins falls **31× → 1.9×**. Share below 5 dB falls
**0.897 → 0.427** (ideal 0.500 — a slight over-correction). The 0.897 figure
independently reproduces A4's separately-measured 0.859, which is a useful
consistency check on two different instruments.

### 3.2 Per-bin results, every contrast (2000 paired trials, bootstrap CI95)

**P13 — SNR-balanced (C1) vs uniform loss (U1), both at `P = 20`.**
Positive = C1 better.

| bin | −10..−5 | −5..0 | 0..5 | 5..10 | 10..15 | 15..20 |
|---|---|---|---|---|---|---|
| Δ (dB) | +0.038 | +0.044 | +0.511 | +1.353 | +1.974 | +2.628 |
| CI95 | [+0.00,+0.08] | [−0.01,+0.11] | [+0.43,+0.59] | [+1.24,+1.44] | [+1.85,+2.06] | [+2.52,+2.77] |

**SNR ≥ 5: +1.902 dB** [+1.823, +1.986]. **SNR < 5: +0.135 dB** [+0.101,
+0.172].

Gap to the unstructured-LS oracle at SNR ≥ 5 falls **+2.897 → +1.000 dB**. In
the top bin C1's median (−11.630 dB) edges past the oracle's (−11.582) — the
oracle is a *perfect-phase, unstructured-LS* ceiling, not a universal bound, and
a learned estimator that exploits channel structure is entitled to pass it.

**P14 — matched-pilot training.** Positive = matched better than the `P = 20`
model evaluated out of distribution.

| bin | −10..−5 | −5..0 | 0..5 | 5..10 | 10..15 | 15..20 | SNR≥5 | SNR<5 | pooled |
|---|---|---|---|---|---|---|---|---|---|
| `P = 10` (C2) | +4.331 | +2.769 | +2.314 | +1.789 | +1.305 | +1.198 | +1.472 | +3.079 | **+2.231** |
| `P = 35` (C3) | +0.339 | +0.655 | +0.776 | +0.765 | +0.660 | +0.706 | +0.728 | +0.559 | +0.635 |

Every bin's CI excludes zero in both rows. The asymmetry is itself a finding:
the `P = 20` model generalizes *upward* in pilot count far better than downward.

**P15, learned half — G1 (C4) vs URformer (C2), both matched-trained at
`P = 10`.** Positive = G1 better.

| bin | −10..−5 | −5..0 | 0..5 | 5..10 | 10..15 | 15..20 |
|---|---|---|---|---|---|---|
| Δ (dB) | −0.020 | −0.190 | −0.321 | −0.214 | +0.515 | +1.293 |

**SNR ≥ 5: +0.498 dB** [+0.405, +0.570]. **SNR < 5: −0.140 dB.** The crossover
moves from ~5 dB (at `P = 20`) to ~10 dB.

**C5 — out-of-distribution path richness.** Trained at `L_k ~ U{3,7}`, evaluated
unchanged at `L_k ~ U{5,10}`; degradation in dB, smaller is better:

| model | U1 | C1 | C2 | **G1 (C4)** | oracle |
|---|---|---|---|---|---|
| degradation | +0.83 | +1.18 | +0.73 | **+0.48** | +0.01 / −0.12 |

Every learned model degrades; the oracle is flat, which confirms the shift is
not making the *problem* harder, only the *learned prior* less apt. **G1
degrades least.** Read with §3.2's P15 row, the structural prior's payoff in
this project is robustness under distribution shift, not in-distribution
accuracy — and that is a different claim from the one the Hankel line was
originally pursuing.

---

## 4. Did P11–P15 hold? Stated plainly

<!-- P11, P12, P15-classical FILLED IN WITH PART B -->

**P13 — CORRECT on the half that mattered, WRONG on the other, favourably.**
Predicted SNR ≥ 5 gain of +1.5 to +2.2 dB (most likely +1.8); measured
**+1.902**. Predicted a low-SNR *cost* of −0.10 to −0.40 dB; measured
**+0.135 dB — a gain.** My stated reason ("some cost is arithmetically
unavoidable — the lowest bin's weight drops ~18×") was wrong: down-weighting
the bins that dominated the gradient did not damage them, because they were
over-served, not merely well-served.

**P14 — CORRECT.** Predicted +1.2 to +2.5 dB (most likely +1.7) at `P = 10`;
measured **+2.231 dB pooled**. The falsifier (margin below +0.5 dB) is nowhere
near.

**P15, learned half — direction RIGHT, magnitude FALSIFIED.** I predicted the
learned HS advantage would *not* grow as `P` falls, at **+0.8 to +1.4 dB on
SNR ≥ 5**, "statistically indistinguishable from the +1.178 dB G1 achieved at
`P = 20`". It measured **+0.498 dB** — it did not grow (direction correct), but
it *shrank*, and 0.498 is outside the interval I wrote down. The pre-registered
falsifier was one-sided (it only fired if the advantage *exceeded* +0.5 dB over
the `P = 20` value), so it did not catch this. **That is a defect in the
pre-registration, not a pass.** A prediction with a stated interval should be
scored against the interval, and this one missed it.

**The documented bias correction was itself mis-applied.** The prereg says: bias
*down* for a structural addition to a trained network, *up* for the network's
own capability. Both P13 and P15 are consistent with the underlying bias being
even stronger than I corrected for — the network out-performed me on its own
capability (P13's low-SNR half) *and* needed the structural prior even less than
I allowed (P15). The correction moved the right direction and not far enough.

---

## 5. Which normalizations collapsed, and which did not

**A failure to collapse is a finding, not a gap.** Recorded either way:

| normalization | collapsed? | evidence |
|---|---|---|
| `K` out of the compression ratio (A1) | **yes** [MATH] | exact cancellation, verified numerically |
| `Δ_HS` re-indexed by `r_eff/cap` (A2) | **yes** [FACT] | zero crossing moves 0.90 → 0.518, monotone |
| `Δ_{GS−EM-GS}` onto `κ` (A3) | **yes** [FACT] | `Δ·κ = 5.32` while κ spans 375×; second family confirms |
| the estimate's own `r_eff` as an index | **no** [FACT] | set by the noise floor (8.91–11.77 across `L = 1…16`), collapses every cell to one point |
| pilot count as a single axis for learned models | **no** [FACT] | P14: efficiency and generalization differ by +2.23 dB at `P = 10`; one curve cannot carry both |

<!-- B1/B2 COLLAPSE ROWS APPENDED WITH PART B -->

---

## 6. Repository state

<!-- FILLED IN AT THE END -->

---

## Files

- `reports/trackD_normalization.md`, `reports/trackD_partA9_normalization.json`
- `reports/trackD_prompt9_prereg.md` (committed standalone, `e24c62a`)
- `reports/trackD_stage5_results.json`, `reports/trackD_stage5_eval.json`
- `reports/trackD_partB9_analysis.json`, `reports/trackD_cost_table.json`
- `scratch/trackD_partA9_normalization.py`, `scratch/trackD_partB9_sweeps.py`,
  `scratch/trackD_partB9_analysis.py`, `scratch/trackD_stage5_eval.py`,
  `scratch/trackD_matched_pilot_points.py`, `scratch/trackD_cost_table.py`
- `trackD_urformer/stage5.py`
- figures: `results/track_d/partB9/fig_collapse_reff.png`,
  `results/track_d/partB9/fig_pilots_three_way.png`
