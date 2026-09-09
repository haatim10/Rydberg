# PROMPT 17 Part B — the seed-1 conditional

**Committed standing alone, before `scratch/p16_score_p27.py` is run and before
any per-seed value is seen.** Deciding this after the scoring is visible is
exactly what pre-registration exists to prevent.

## The confound

Recorded in `reports/p16/PREREG_P27.md` before the four Tier 0.5 runs started:

- **Seed 1** is the published pair. Its unbalanced arm,
  `results/track_d/stage2/B3_80k_13ep`, was trained by `stage2.py`; its
  balanced arm, `results/track_d/stage5/C1_snr_balanced_P20`, by `stage5.py`.
- **Seeds 2 and 3** use one driver for both arms
  (`scratch/p16_tier05.py`).

The two drivers are the same procedure — same optimiser (Adam, lr 1e-3), same
cosine schedule over 13 epochs, same gradient clipping at 1.0, same one-SE
epoch selection (`stage5.py:53` imports `select_epoch` from `stage2.py`), and
`weighted_nmse_loss(w=None)` is exactly `nmse_loss`. But "same procedure" is an
argument, not a measurement, and seed 1 is the one point where driver and seed
vary together.

## The rule

> **If seed 1 is the outlier among the three — furthest from the mean of the
> other two, in either direction — the driver difference and the seed are
> confounded, and the three-seed spread is not interpretable as a seed spread.
> In that case, seed 1 is re-run under the unified driver (2 runs, ~1.3 h) and
> the contrast is re-scored before any spread is quoted.**
>
> **If seed 1 is not the outlier, the driver difference is recorded as a
> limitation and no re-run is needed.**

## Operationalised, so it cannot be argued about afterwards

Let the three SNR ≥ 5 dB gains be `g1, g2, g3`. Seed 1 is the outlier iff

```
|g1 - mean(g2, g3)|  >  max( |g2 - mean(g1, g3)| , |g3 - mean(g1, g2)| )
```

That is: the deviation of each seed from the mean of the *other two* is
computed, and seed 1 is the outlier iff its deviation is the largest of the
three. Ties go to "not the outlier", since a tie is not evidence of a confound.

This is deliberately a **rank** test rather than a threshold. With three points
there is always exactly one largest deviation, so seed 1 is the outlier with
prior probability 1/3 under the null that the driver makes no difference. The
rule therefore fires a third of the time by chance, and firing is not by itself
evidence of a driver effect — it is a trigger for the cheap check that removes
the ambiguity, which is the only thing two extra runs can buy here.

## What is reported either way

- Whether the rule fired, stated before the spread is quoted.
- If it fired: the re-run, the re-scored spread, and both spreads side by side,
  so the size of the driver effect is visible rather than assumed away.
- If it did not fire: the driver difference appears in Paper 2's limitations as
  a named, unquantified caveat on seed 1 — not as a claim that it is
  negligible, which nothing here establishes.

## What this rule does not do

It does not establish that the two drivers agree. Three seeds cannot support
that, and a rank test on three points certainly cannot. It only prevents the
specific failure of quoting a seed spread that is partly a driver spread
without knowing which.
