# Work log — Paper 1 finished, Paper 2 blocked

Covers everything since PROMPT 14, ending at commit `23d4d00`. Working tree is
clean, local and remote are in sync, and nothing is running.

---

## 1. Where Paper 2 actually stands

**Not done, and further from done than before**, because the finding it is
built on did not survive checking.

The draft is `paper/spl2/haatim_structural_priors_evaluation.pdf` — 4 pages,
titled *"On the Evaluation of Structural Priors in Unrolled Channel Estimation
for Rydberg Atomic Receivers"*. It was last touched only incidentally, by the
PROMPT 14 bibliography repair. It has never been restructured.

### The claim the draft is built on is dead

The draft says, in four places:

> *"Train both arms adequately and a $+1.209$ dB structural advantage falls by
> a factor of fifteen, to $+0.078$ dB."* (line 43)
>
> *"…a fifteenfold collapse (Fig. 2)."* (line 192)
>
> *"…under a matched-adequacy control the effect falls fifteenfold, from
> $+1.209$ to $+0.078$ dB."* (line 336)

The word "fifteen" appears 3 times, `1.209` and `0.078` four times each.

PROMPT 15 was written specifically because that framing is unsupported, and it
supersedes PROMPT 13: *"It was written to restructure Paper 2 around a
fifteenfold collapse that Part C does not support. Its instructions would now
produce a false paper."*

### Why the supporting evidence collapsed too

PROMPT 12 Part C was supposed to settle whether the collapse survives across
seeds. It reported that it does not. **That report has since been withdrawn**,
because Part C itself was wrong — see §3 below. So the draft's central claim is
unsupported, *and* the work that was meant to replace it is invalid.

### PROMPT 15 status, part by part

| Part | What it asked for | Status |
|---|---|---|
| **A** | diagnose non-reproducible evaluation, fix, re-score, blast radius, Paper 1 exposure check | **DONE** — gate passed |
| **B** | B1 projection-disabled at inference; B2 three-design per-bin table | **BLOCKED** — reads from mis-trained arms |
| **C** | pre-register P24–P26, retrain three tiers with more seeds | **BLOCKED** — needs retraining first |
| **D** | write `docs/paper2-thesis.md` | **NOT STARTED** — needs no compute, offered, awaiting your go-ahead |

---

## 2. What was done recently

### PROMPT 14 — Paper 1 closed out (`612ac12` … `99a27e0`)

- **Aperture sweep at the principal operator.** 30 operating points,
  N ∈ {8,16,32} × P ∈ {10,30} × 6 SNR × 400 paired trials, ~2.6 h on four
  shards. Scored against P22, which was pre-registered at `7c6708a` *before*
  the sweep ran.
- **P22 FAILED**, reported as such: N=8 (−0.254) and N=32 (+2.934) landed
  inside their predicted bands, N=16 (+0.966) missed [+0.60, +0.95] by
  **0.016 dB**.
- **Two findings that changed what the letter claims.**
  1. The predicted sign reversal at N=8 is a **ratio-of-sums phenomenon**.
     Under the paired per-trial median — the letter's own primary statistic —
     N=8 is **+0.225 dB**: exactly +0.000 at nine of twelve points, positive
     and significant at three, negative and significant at none, because the
     selector abstains in 42% of trials. §IV-B now reports both poolings and
     says the prediction is confirmed on the statistic a bound governs and is
     invisible to the one otherwise called primary.
  2. The EM-GS flatness figure depends on the pooling too. Averaging the two
     pilot counts gives 0.033 dB because they move in opposite directions; per
     pilot count it is **0.163 dB** (P=10) and 0.097 dB (P=30). The letter
     quotes 0.163 — the larger, less flattering figure.
- **Configuration bookkeeping restored.** Table I now names the four
  configurations actually used and every number in §IV carries its label. This
  exposed that the crossing triple `0.588 / 0.518 / 0.544` spans **two**
  configurations, and that only the N=32 crossing is bracketed on both sides.
- **A correction to the brief itself.** PROMPT 14 B1 stated the §IV-D/§IV-E
  results use `n_cz = 1`. They do not — `hs_gs_auto` is called without
  `cadzow_iter` and the default is 4. All four configurations use `n_cz = 4`.
  Recorded in `docs/open-todos.md`.
- **Citations repaired**, `\todo` macro removed entirely so a stray marker is a
  build error, and `docs/open-todos.md` records everything deliberately left
  out rather than silently dropped.
- **Build gate** `scripts/submission_gate.sh` — eight checks, all passing.

### PROMPT 15 Part A — the blocking gate (`c2c0583`, `f690220`, `88752e4`)

The brief's premise was that evaluation is nondeterministic. **It is not.**
Every leg of the discriminating test came back clean:

| test | result |
|---|---|
| `stage4.evaluate()` twice in one process | bitwise identical |
| `p12_partC_eval.eval_models()` twice | bitwise identical |
| the two code paths against each other | bitwise identical, per trial |
| stored test-set SNR draws vs today's | bitwise identical, all 2000 |

Also ruled out by direct check, not assertion: code drift (`git diff` over
`trackD_urformer/` and `rydberg_sim/` since the checkpoint commit is **empty**),
environment drift (`requirements.txt` pins the exact versions installed), and
partial checkpoint loading (all 480 `state_dict` keys present and overwritten).

**The real cause was mine.** `TrackDConfig().train.init` defaults to
`"random"`; every stage script overrides it to `"spectral"`. Two of my PROMPT 12
scripts did not:

- `scratch/p12_partC_eval.py` — *evaluated* with a random `G0`;
- `scratch/p12_partC.py` — ***trained*** the seed-2, seed-3, balanced and log
  arms with a random `G0`.

The checkpoints record it: stage-4 arms carry `train.init='spectral'`, every
`results/p12/partC/` arm carries `'random'`.

It survived scrutiny because `make_initial_G("random", ..., seed=trial)` is
seeded per trial, so the wrong evaluation was *perfectly deterministic* —
bitwise reproducible across processes and across two independent code paths.
Determinism and correctness are different properties, and every test applied
until then, including the ones the brief specified, tested only the first.

---

## 3. Three conclusions withdrawn

| quantity | reported | re-scored | Δ |
|---|---|---|---|
| P19 Δ_H seed 1 | +0.690 | **+0.078** | −0.612 |
| P20 Δ_H, SNR ≥ 5 | +0.395 | **+0.174** | −0.221 |
| P21 log − balanced | +0.584 | **+0.343** | −0.241 |

1. **"Stage 4 does not reproduce its own stored rows" was my bug.** Seed 1
   re-scores to **+0.0778 dB** — exactly the published stage-4 value. The
   repository was always right.
2. **The P20 TENSION branch does not fire.** +0.174 dB is *inside* the
   pre-registered [−0.10, +0.35] band. P20 still fails its per-bin threshold
   (top bin +0.919 > +0.50), but the claim that the recommended loss restores
   the prior's value is not supported.
3. **P21 held, it did not fail.** +0.343 dB is inside the ±0.5 dB equivalence
   margin, so the log loss does *not* beat per-bin reweighting by +0.584 dB.
   The pre-registration's committed "yes" was right.

**None of the re-scored values may be used either** — the arms behind them were
trained with the wrong initialiser. They are in `reports/p15/rescored.csv` with
a validity column, recorded to size the error, not as results.

### What survives for Paper 2

- **The gradient-share diagnostic.** 89.7% of the gradient comes from below
  5 dB (recomputed as `sum(grad_share[:3]) = 0.8966`), span **30.0**
  (`0.4653/0.0155`). These measure a *loss function*, not a trained model, so
  the initialiser bug cannot touch them. Note the manuscript's `31` is wrong;
  30 is correct.
- **`+1.902` and `+2.231` dB** — stage-5 arms, and `stage5.py:277` sets the
  initialiser; every stage-5 checkpoint records `train.init='spectral'`.
- **`+1.209`, `+0.078`, `+1.309` dB** — stage-3/stage-4 rows, produced by the
  stage scripts. Stage 4's reproduce bitwise today.

The numbers are fine. It is the *interpretation* — that 1.209 → 0.078 is a
fifteenfold collapse rather than one draw from an unmeasured distribution —
that has no support, because the experiment meant to measure that distribution
was invalid.

---

## 4. Fixes landed

- Both scripts now set `init="spectral"`, with the trap documented in place.
- `tests/test_trackd_eval_reproducibility.py` pins six properties: the config
  default is still `"random"` (so the override stays necessary); `eval_models`
  hands the model a spectral `G0`, checked by spying on the dataset rather than
  by reading source; `run_cfg` sets it for training; `evaluate()` is
  deterministic in-process; the published stage-4 rows reproduce from `best.pt`
  bitwise; and the stage-4 checkpoints record `train.init='spectral'`.
  **All pass.**
- `reports/p12/PART_C.md` carries a WITHDRAWN notice.
- Full diagnosis in `reports/p15/PART_A.md`.

**Paper 1 is unexposed**, verified rather than assumed: every `% src:` comment
in the letter resolves to a classical results file, its only placement number
is the classical B5 wash (+0.003 dB), and the trained-loop contrast is
deliberately uncited (`docs/open-todos.md` item 3).

---

## 5. Paper 1 — done

Two versions, both submission-ready, both 5 pages, all eight gates passing:

| file | voice |
|---|---|
| `paper/paper1/haatim_hsgs_letter.tex` | original |
| `paper/paper1/haatim_hsgs_letter_restyled.tex` | register of the three IEEE letters supplied |

The restyle changed voice only — verified mechanically by comparing the
multiset of decimal numbers in the two PDFs: **131 each, none added, dropped or
changed in multiplicity.**

Recent cosmetic passes: Fig. 2 decluttered (four ways of encoding a series
reduced to one, single legend); the restyled abstract rewritten from one
235-word paragraph with 45- and 55-word sentences into fourteen shorter ones;
Fig. 1(b) decluttered (five text objects down to two). A stale caption in the
frozen letter was also corrected — it still claimed the secondary axis span was
fixed at ±3 dB, which stopped being true when that axis was widened.

---

## 6. What is waiting on you

1. **Retraining budget for Paper 2.** The three PROMPT 15 Part C tiers
   (mixed-SNR, balanced, log) each need retraining with the correct
   initialiser before P24–P26 mean anything. That is roughly double what the
   brief budgeted. Parts B and C are blocked until then.
2. **`docs/paper2-thesis.md` (PROMPT 15 Part D).** Needs no compute, and is
   meant to be committed *before* the numbers land so the framing cannot be
   retrofitted to them. One edit is needed against the brief's version: the
   "evaluation path" nuisance source is no longer hypothetical — it is
   measured, and the initialiser is the example. Offered, not started.
