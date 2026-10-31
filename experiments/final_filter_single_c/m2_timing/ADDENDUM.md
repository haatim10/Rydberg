---

# ADDENDUM (post hoc, 31 October 2026): M2 timing and arm C3

**Everything in this addendum was planned after the main test results were seen.**
It does not change the frozen results, choices or verdicts above. Pre-registration
for the addendum: [m2_timing/configs/followup_preregistration.json](m2_timing/configs/followup_preregistration.json),
committed before any follow-up data. Code and results: [m2_timing/](m2_timing/).

## A1. Runtime limit and method

- **The limit is the original pre-registered one:** absolute p90 ≤ 10.74 ms, one CPU
  thread, double precision. The 1.02× arm-A ratio in Section 4 was only the
  fallback for the cloud machine, where arm A itself exceeded 10.74 ms.
- **Method:** arms A, C2 and C3 run interleaved in one process. Each world and fit
  calls all three back to back on the same inputs, with the order rotated per
  call. There is a warm-up and nothing else runs. Each arm makes 1,377 calls
  (459 grid worlds × 3 fits). Online outputs are checked to equal the offline
  arm computation exactly at the first world of every setting.
- **Scopes.** The follow-up request calls the second scope the "30–60 dB target
  range" (all RSR). The main pre-registration defined the target range as RSR 5
  above 30 dB. Both are reported. The pass/fail columns apply the limit to the
  full grid together with 30–60 dB, and separately to RSR 5 above 30 dB.

## A2. Apple M2 timing

**Pending: not yet run on the Apple M2.** This session runs on a cloud x86 machine (Intel Xeon, 4 cores) and could not reach the M2. Run on the M2, from the repository root with nothing else running:

```bash
.venv/bin/python -m experiments.final_filter_single_c.m2_timing.timing_interleaved --tag m2
```

Then regenerate this addendum and the report:

```bash
.venv/bin/python -m experiments.final_filter_single_c.m2_timing.addendum
```

```bash
.venv/bin/python -m experiments.final_filter_single_c.report
```

Until then, **no pass/fail against 10.74 ms exists for C2 or C3 on the reference hardware.**

## A3. Cloud diagnostic (not an M2 result, no pass/fail)

The same interleaved script on the cloud machine (Intel(R) Xeon(R) Processor @ 2.10GHz):

| Arm | Full grid p90 / max ms | 30–60 dB (all RSR) p90 / max ms | RSR 5 > 30 dB p90 / max ms | p90 ratio vs A (full / 30–60 / RSR 5) |
| --- | --- | --- | --- | --- |
| A | 9.14 / 17.7 | 10.68 / 17.7 | 10.33 / 13.9 | — |
| C2 | 9.11 / 16.0 | 10.83 / 16.0 | 10.93 / 15.0 | 0.997 / 1.014 / 1.058 |
| C3 | 9.23 / 20.3 | 10.86 / 20.3 | 10.89 / 15.5 | 1.011 / 1.017 / 1.054 |

Interleaving removes most of the drift between the two separate timing windows in
Section 6. The paired per-call ratio against A has a median of
1.0035 for C2 and
1.0032 for C3. On this machine A's p90 is already close to
10.74 ms at 30–60 dB, so these absolute values say nothing about the M2.

## A4. Arm C3: filter only where the EM tail is nonzero

C3 is C2 (c = 4, β = 1, nominal SNR > 30 dB), filtering only where the routed
EM-tail length is > 0. That means reference key 5 from 40 dB, reference keys 12/20
from 32.5 dB, and never at 31.25 dB. It was evaluated on fresh draws (seed
2026103121): 396 cells × 128 channels × 3 fits, with the main study's metrics and
20,000-draw block bootstrap. Arm A again reproduced the frozen receiver exactly
in every cell.

Mean-setting NMSE (dB) and differences:

| Scope | A | C2 | C3 | C2 − A [95% CI] | C3 − A [95% CI] | C3 − C2 [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| Full grid (153) | -38.71 | -39.42 | -39.25 | -0.72 [-0.73, -0.70] | -0.54 [-0.55, -0.53] | +0.17 [+0.16, +0.19] |
| 30–60 dB (81) | -50.31 | -51.66 | -51.33 | -1.35 [-1.39, -1.32] | -1.02 [-1.04, -1.01] | +0.33 [+0.30, +0.35] |
| Off-grid SNR, grid RSR (45) | -45.71 | -46.82 | -46.18 | -1.11 [-1.17, -1.05] | -0.47 [-0.48, -0.46] | +0.64 [+0.58, +0.69] |
| 31.25 dB, all RSR (18) | -35.96 | -36.64 | -35.96 | -0.68 [-0.77, -0.60] | +0.00 [+0.00, +0.00] | +0.68 [+0.60, +0.77] |
| RSR 0 (66) | -30.11 | -32.60 | -31.36 | -2.48 [-2.56, -2.42] | -1.25 [-1.29, -1.21] | +1.24 [+1.17, +1.30] |
| RSR 8 (66) | -38.68 | -40.96 | -40.09 | -2.29 [-2.36, -2.22] | -1.41 [-1.43, -1.38] | +0.88 [+0.82, +0.94] |
| RSR 16 (66) | -40.96 | -41.02 | -41.06 | -0.06 [-0.07, -0.05] | -0.10 [-0.11, -0.09] | -0.04 [-0.05, -0.04] |

**Q1. Does C3 remove the > 1 dB losses at 31.25 dB? Yes.**
- C3 is identical to the table at 31.25 dB (difference exactly 0).
- On these fresh draws, C2 was > 1 dB worse than A in 4 cells at 31.25 dB
  (`L4_rsr12_snr31.25`, `random3to7_rsr12_snr31.25`, `random3to7_rsr20_snr31.25`, `random3to7_rsr16_snr31.25`).
- C3 has none.

| Cell | EM tail | C2 − A dB | C3 − A dB |
| --- | --- | --- | --- |
| `L2_rsr12_snr31.25` | 0 | +0.72 | +0.00 |
| `L4_rsr12_snr31.25` | 0 | +1.10 | +0.00 |
| `random3to7_rsr12_snr31.25` | 0 | +1.00 | +0.00 |
| `L2_rsr20_snr31.25` | 0 | +0.73 | +0.00 |
| `L4_rsr20_snr31.25` | 0 | +0.91 | +0.00 |
| `random3to7_rsr20_snr31.25` | 0 | +1.05 | +0.00 |
| `L2_rsr16_snr31.25` | 0 | +0.80 | +0.00 |
| `L4_rsr16_snr31.25` | 0 | +0.97 | +0.00 |
| `random3to7_rsr16_snr31.25` | 0 | +1.04 | +0.00 |

**Q2. Does C3 keep C2's gains elsewhere? No.**
- C3 is worse than C2 by more than 0.3 dB in 63 cells.
- All of them are at reference key 5 (RSR 0, 5 and 8) between 31.25 and 38.75 dB.
  There the receiver's EM tail is zero, so C3 stops filtering where C2 gained most.
- The worst is `L2_rsr0_snr38.75`, where C3 is +4.96 dB worse than C2.
- Elsewhere C3 ≈ C2. At RSR 16, C3 is slightly better because it skips 31.25 dB.

**Against the table (main-study limits, descriptive because post hoc):**
- C3: L1 -0.54 dB full grid and -1.02 dB at 30–60 dB.
  L2 worst grid setting +0.08 dB (`random3to7_rsr20_snr32.5`).
  L3: 0 settings with BER harm.
- On the same fresh draws, C2's worst grid setting is +0.08 dB, with
  0 BER-harm settings.
- BER is identical across A, C2 and C3 in every scope, and zero at 30–60 dB.

![C2 and C3 minus A](m2_timing/figures/followup_c3_vs_c2.png)

**Reading.**
- The 31.25 dB losses do not come from the filter strength. They come from the
  frozen receiver's routing at that point: global start, zero-length tail, and a
  30 dB table key. Removing the filter wherever the tail is zero fixes 31.25 dB at
  RSR 12/16/20, but also removes the large RSR 5 gains between 32.5 and 38.75 dB.
- On these draws, C2 is ahead of C3 by 0.17 dB over the full grid and
  0.33 dB at 30–60 dB.
- So whether C2 replaces the table still depends on the M2 timing (A2). C3 is the
  more conservative variant if you want no setting to lose to the table at all.
- A narrower variant that skips the filter only at the 31.25 dB mismatch, or a fix
  to that routing quirk, was not tested.
