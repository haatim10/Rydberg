# PROMPT 17 Part A — P27 scored

Source: `reports/p16/p27_score.json`, produced by `scratch/p16_score_p27.py`
with thresholds transcribed from `reports/p16/PREREG_P27.md` (committed
`b4a1dc8`, before any run started).

---

## A1. Scorer correctness check

**Seed 1 reproduces `+1.9024` dB against the published `+1.902`.** [FACT]

The scorer reuses `trackD_stage5_eval.py`'s evaluation path — same
`TrackDDataset` worlds, `P = 20`, spectral initialiser, `n_test = 2000`, same
`by_bin` edges and sign convention. The agreement is exact to the published
precision, so the scorer is reading the right worlds through the right path and
seeds 2 and 3 can be read.

---

## A2. The three seeds

Contrast is `U1` (uniform loss) − `C1` (balanced loss); **positive means the
balanced loss is better**.

| seed | gain at SNR ≥ 5 (dB) | CI over **test realisations** | driver |
|---|---|---|---|
| 1 | **+1.9024** | [+1.823, +1.986] | `stage2.py` + `stage5.py` (mixed) |
| 2 | **+2.2427** | [+2.158, +2.322] | `p16_tier05.py` (unified) |
| 3 | **+2.3691** | [+2.257, +2.452] | `p16_tier05.py` (unified) |

| across-seed quantity | value |
|---|---|
| three-seed mean | **+2.1714** dB |
| `seed_spread_sd_db` | **0.2414** dB |
| `seed_spread_range_db` | **0.4667** dB |
| `mean − 2·SD` | **+1.6887** dB |

**These are two different quantities and are never pooled.** The per-seed
intervals are bootstrap CIs over the 2000 paired test realisations. The SD and
range are over seeds. Conflating them is the error that produced the withdrawn
P19.

**Three seeds give two degrees of freedom.** The SD carries large sampling
uncertainty — the point estimate could be off by a factor of two with nothing
wrong — so any statement Paper 2 makes from it must say it was evaluated on
three seeds.

### Verdicts

**P27a HELD.** SD `0.2414` against a threshold of `0.45`. The pre-registered
point estimate was `0.25`.

**P27b HELD**, on all five clauses:

| clause | threshold | measured |
|---|---|---|
| three-seed mean | ≥ +1.40 | **+2.171** |
| every seed | > +1.00 | min **+1.902** |
| per-bin means monotone | non-decreasing | **yes** |
| top bin | ≥ +2.00 | **+3.120** |
| neither low bin | ≥ −0.10 | **+0.052, +0.202** |

**Separation: `mean − 2·SD = +1.689 > 0`.** Under the rule fixed before the
run, the headline is separated from no-effect and may be stated with the seed
spread alongside it.

**But see Part B — the spread is not yet quotable.**

---

## A3. The balanced loss wins in the bins it was weighted *away* from

Per-bin contrast, `U1 − C1`, positive = balanced better. `*` marks a bootstrap
CI over test realisations that excludes zero.

| bin (dB) | seed 1 | seed 2 | seed 3 | 3-seed mean |
|---|---|---|---|---|
| [−10, −5) | +0.038 * | +0.071 * | +0.048 * | **+0.052** |
| [−5, 0) | +0.044 | +0.390 * | +0.170 * | **+0.202** |
| [0, 5) | +0.511 * | +0.898 * | +0.665 * | **+0.692** |
| [5, 10) | +1.353 * | +1.618 * | +1.446 * | **+1.472** |
| [10, 15) | +1.974 * | +2.242 * | +2.352 * | **+2.189** |
| [15, 20) | +2.628 * | +3.038 * | +3.694 * | **+3.120** |

**The balanced arm is better in every bin, for every seed** — 18 of 18 cells
positive, 17 of 18 with a CI excluding zero. The single exception is seed 1's
`[−5, 0)` bin at `+0.044`, CI `[−0.011, +0.105]`. [FACT]

**This is the non-obvious part.** The two lowest bins carry weights of `0.056`
and `0.111` — the balanced loss deliberately down-weights them by factors of
18 and 9 relative to unit mean. It is better there anyway.

So the fix is **not** trading low-SNR accuracy for high-SNR accuracy, which is
what a reweighting is normally suspected of doing. It improves the quantity it
was weighted away from. [FACT]

The natural reading, offered as a hypothesis and not established here: under
the conventional loss the gradient was so dominated by the low-SNR tail
(89.7% below 5 dB) that the network was underfitting *everywhere*, spending its
capacity on samples where little is recoverable. Rebalancing does not move
performance from one regime to another; it stops wasting it. [HYP]

This is a sentence Paper 2 can carry, and it cost one evaluation pass.

---

## What is settled and what is not

**Settled.** The scorer is correct (A1). The balanced loss improves every bin
for every seed, including the down-weighted ones (A3). P27a and P27b both held
against thresholds fixed before the runs.

**Not settled.** The seed spread itself, because the Part B conditional fired —
seed 1 is the outlier and its driver and its seed vary together. The numbers
above are reported because the brief asks for them; **the spread is not quoted
as a seed spread** until the seed-1 re-run under the unified driver lands.
