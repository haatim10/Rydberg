# What the frozen receiver feeds the final filter

Generated from the receiver's own routing code by `policy_trace.py`; the full
per-setting table, including off-grid points, is `results/policy_trace.json`.
Ties in the nearest-key lookup were checked on real generated worlds; they always
resolve to the lower key (64/64 worlds per point).

## Where the policy lives and who applies it

| Piece | File | What it does |
| --- | --- | --- |
| Table of filter choices | `results/tcom_filter_continuation_20261019/policy.json` → `repair_choices` | Named choice per reference key and SNR key |
| Choice → (c, β) | `tcom_start_budget_20261020/study.py: filter_config, filter_for, finish` | `shrink_c{c}_beta{β}` → that (c, β). **Every other choice, including `warm5` and `warm100`, → c = 0 (no filter)** |
| Filter itself | `tcom_incumbent_guard_20261019/shrink.py: apply` | `τ_k = c sqrt(2v Re[(SSᴴ)⁻¹]_kk max(m,q))`, smooth Hankel shrinkage, β mix |
| Routing, EM tail, call of `finish` | `tcom_hybrid_20261020/inference.py: plans, hybrid` | ≤ 20 dB: adaptive start, `finish` **not called**. Above 20 dB: adaptive start up to the 30 dB crossover, global start above it (or at ≥ 40 dB). Then the EM tail from `proposal_policy.tails`, then `finish` |
| EM-tail lengths | `results/tcom_hybrid_sensitivity_20261020/protocol.json` → `proposal_policy.tails` | 0/10/20/40 exact EM steps per reference/SNR key |
| Recovery check | `tcom_reliability_20261021/inference.py: from_base, fallback` | After `finish`. Only at reference key 5 and nominal SNR > 30 dB |

Keys come from observables: reference key = nearest of {5, 12, 20} to
`10 log10 mean|B|²`; SNR key = nearest grid SNR to `10 log10(K/v)`. The learned
stages hand over to the high-SNR path at nominal SNR **> 30 dB** (`crossover_db = 30`),
where the global fixed-schedule start replaces the adaptive one.

## Confirmation of the stated policy

| Reference | Stated | Found |
| --- | --- | --- |
| RSR 12 | off ≤ 30 dB, c = 2 at 32.5–35, c = 4 from 37.5 (β = 1) | **Confirmed** |
| RSR 20 | off ≤ 30 dB, c = 2 at 32.5, c = 4 from 35 (β = 1) | **Confirmed** |
| RSR 5 | never filtered; warm EM5 at 35 dB, warm EM100 from 37.5 dB | **Never filtered: confirmed.** The `warm5`/`warm100` entries are in the file but **are not used** by the current receiver. `filter_config` maps them to "no filter". The EM tail at RSR 5 comes from `proposal_policy.tails`: 0 steps at 32.5–37.5 dB and 40 steps from 40 dB |

The `warm*` entries were applied by an earlier receiver
(`tcom_transition_repair_20261019`, `tcom_incumbent_guard_20261019`), not by the
hybrid/reliability receiver that the paper reports.

## Exact input to the final filter, per setting

| Setting | Final-filter input | Table filter |
| --- | --- | --- |
| SNR ≤ 20 dB (all RSR) | Adaptive 10-stage output. `finish` is not called | none |
| 20 < SNR ≤ 30 dB (all RSR) | Adaptive 10-stage output + 0 EM steps | none (c = 0) |
| RSR 5, 32.5–37.5 dB | Global 10-stage output + 0 EM steps; may be replaced by the recovery estimate | none |
| RSR 5, 40–60 dB | Global 10-stage output + 40 EM steps; may be replaced by the recovery estimate | none |
| RSR 12, 32.5–35 dB | Global + 10 EM steps | c = 2 |
| RSR 12, 37.5–45 dB | Global + 20 EM steps | c = 4 |
| RSR 12, 50–60 dB | Global + 40 EM steps | c = 4 |
| RSR 20, 32.5 dB | Global + 10 EM steps | c = 2 |
| RSR 20, 35–60 dB | Global + 40 EM steps | c = 4 |

The recovery estimate is an algebraic start, 20 EM steps and `finish`. At
reference key 5 `finish` is the identity, so the replacement is unfiltered.

**Recovery and the filter.** The recovery check runs **after** the final filter
and scores the filtered incumbent. At reference key 5, the only place it can fire,
the table filter is off. So the receiver's recovery decisions and alternatives
do not depend on the filter. This study keeps them fixed at the frozen receiver's
decisions. Each arm's filter is applied to the estimate the frozen receiver
selected: the incumbent before filtering, or the accepted recovery estimate.
That is what "everything upstream bit-identical" means here.

## Off-grid points

| Point | Keys used | Consequence |
| --- | --- | --- |
| SNR 31.25 dB | SNR key 30, but global route (31.25 > 30) | Family mismatch → zero-step tail; **no filter, even at RSR 12/20** |
| SNR 33.75 / 36.25 / 38.75 / 47.5 dB | 32.5 / 35 / 37.5 / 45 | Tail and filter of the lower key |
| RSR 8 dB, RSR 0 dB | Reference key 5 | RSR 5 behaviour: never filtered; recovery eligible above 30 dB |
| RSR 16 dB | Reference key 12 (tie → lower) | RSR 12 behaviour |

## High-SNR route versus filtered settings (for arm C2)

The high-SNR route (nominal SNR > 30 dB) covers **all** settings the table filters,
plus RSR 5 (and 0, 8) above 30 dB and SNR 31.25 dB at any reference, where the
table does not filter. Inside that route, RSR 5 at 32.5–37.5 dB (and 31.25 dB
everywhere) uses a zero-length EM tail.
