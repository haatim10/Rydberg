# Master paper audit

Audit of `paper/master/haatim_hsgs_master.tex` → `paper/master/haatim_hsgs_master.pdf`.

Compiled 2026-09-05. `pdflatex` ×3 + `bibtex`. **13 pages, 0 undefined
references, 0 LaTeX errors, 0 overfull boxes.** 19 sections, 9 figures,
4 tables, 3 algorithms, 14 cited references, 38 `% SOURCE` provenance comments.

**No simulation was rerun for this document.** Every number below was
recomputed from already-stored `.npz` / `.json` / `.csv` artifacts, or read
directly from them. Two figures (`fig6_gain_vs_raw_L`, `fig9_k_invariance`)
were rendered by `scratch/master_extra_figures.py` from stored results; the
other seven were copied from existing figure directories.

---

## A. What this document is, and how it differs from the submission version

| | `paper/merged/haatim_hsgs_bounds_and_effective_rank.tex` | `paper/master/haatim_hsgs_master.tex` |
|---|---|---|
| Purpose | IEEE SPL submission | complete research record for advisor review |
| Length | 5 pages (hard limit) | 13 pages (no limit) |
| Class | `IEEEtran` letter | `IEEEtran` journal, two-column |
| Figures | 4 | 9 |
| Tables | 2 | 4 |
| Algorithms | 0 (prose) | 3 (full pseudocode) |
| Derivations | stated | Hankel lifting, rank capacity, rank-one Fisher, tangent-space CCRB all carried in full |
| Negative results | the two that fit | all of them, in a dedicated §XVIII with 19 entries |
| Experiment families | B3 + TD | B3 + EXT + TD, each labelled at every use |

**The submission version was not modified by this work.** It remains at
`paper/merged/` exactly as audited previously.

---

## B. Data hygiene: the three experiment families

The load-bearing rule for this document is that the three result stores are
**different experiments** and are never pooled or cross-quoted. Table II of the
manuscript states all three; the file header restates them; every numeric claim
carries a `% SOURCE` comment naming its store.

| Family | Store | Driver | Configuration |
|---|---|---|---|
| **B3** (principal) | `results/track_b/b3/` | `scripts/plot_paper.py` | `N∈{8,16,32}` × `P∈{10,30}` × SNR `{-5,0,5,10,15,20}`; 400 trials/point; RSR 12 dB, `T=50`, `n_cz=4` |
| **EXT** (diagnostic) | `trackB_hankel_emgs/results/` | `figure_array_size_comparison.py` | `P=30` only; 7 SNR points incl. −10 dB; 600/400/200 trials at `N=8/16/32` |
| **TD** (effective rank) | `results/track_d/partB9/` | `scratch/trackD_partB9_*.py` | `N∈{16,32,64}`, `P=20`, SNR ~ `U[-10,20]`, `n_cz=1`, `T=100`, adaptive order |

The B3 and EXT families measure the same quantity (aperture gain) and disagree,
because they are different experiments:

| | B3 (12 op. points/N) | EXT (7 SNR points, P=30) |
|---|---|---|
| Aperture gain `N=8/16/32` | `−0.192 / +0.780 / +2.851` dB | `+0.029 / +0.812 / +2.452` dB |
| EM-GS flatness in `N` | `0.0124` dB | `0.0362` dB |

Both appear in the manuscript (Table IV rows tagged **B3** and **EXT**, and
§IX-C), with the text stating explicitly that the EXT values *"are not the B3
numbers and should not be quoted as such."* The old draft `paper/hsgs.tex`
itself contained both — its Fig. 2 was B3 and its Fig. 5 was EXT — which is the
origin of the discrepancy.

**Result: no cross-configuration contamination found in the final text**, after
the two corrections in §D below.

---

## C. Headline numbers — every one re-verified against its store

All recomputed this session. ✓ = reproduces exactly.

### Family B3 (`results/track_b/b3/*.npz`, ratio-of-sums pooling per point)

| Claim | Manuscript | Recomputed | |
|---|---|---|---|
| Aperture gain `N=8/16/32` (mean over 12 pts) | `−0.19/+0.78/+2.85` | `−0.192/+0.780/+2.851` | ✓ |
| EM-GS flatness in `N` | `0.012` dB | `0.0124` dB | ✓ |
| Win rate `N=8/16/32` | `33.5/74.0/95.3 %` | `33.5/74.0/95.3 %` | ✓ |
| Constraint active `N=8/16/32` | `57.5/92.7/99.6 %` | `57.5/92.7/99.6 %` | ✓ |
| Projection inactive at `N=8` | `42.5 %` | `42.5 %` | ✓ |
| `N=8` gain at −5 dB, `P=30`/`P=10` | `+0.78`/`+1.47` dB | `+0.78`/`+1.47` dB | ✓ |
| `N=8` worst high-SNR gain | `−2.23` dB | `−2.23` dB | ✓ |
| HS-GS over EM-GS, `N=32`,`P=30` | `+2.27` to `+3.04` dB | `2.268–3.039` dB | ✓ |

Both pooling conventions were tested; the manuscript's stated rule (ratio-of-sums
within a point, then mean across points) is the one that reproduces. Win rate and
activation are mean-of-per-point rates, not trial-pooled rates — pooling instead
gives `56.9/91.5/99.6 %`, which is *not* what the manuscript claims, confirming
the stated convention is the operative one.

### Bounds (`results/track_b/crlb.json`, `constrained_crlb.json`; `N=32, P=30`)

| Claim | Manuscript | Recomputed | |
|---|---|---|---|
| EM-GS vs rank-1 unconstrained CRLB, SNR ≥ 5 | within `0.051` dB | max `0.0513` dB | ✓ |
| Same, at −5 / 0 dB | `0.295` / `0.237` dB | `0.2947` / `0.2372` dB | ✓ |
| CCRB below unconstrained bound | `7.05–7.11` dB | `7.045–7.106` dB | ✓ |
| Rank-two construction error | `0.17–4.64` dB too low | `0.174–4.642` dB over 36 pts | ✓ |
| HS-GS above CCRB | `34` of `36` | `34/36` | ✓ |
| HS-GS below unconstrained CRLB | `26` of `36`, up to `6.94` dB | `26/36`, `6.94` dB | ✓ |

### Family EXT (`trackB_hankel_emgs/results/*.csv`, `summary.json`)

| Claim | Manuscript | Recomputed | |
|---|---|---|---|
| Aperture gap `N=8/16/32` | `0.03/0.81/2.45` dB | `0.0288/0.8123/2.4524` | ✓ |
| EM-GS flatness (EXT) | `0.036` dB | `0.0362` dB | ✓ |
| Path-count decay `L=2→16` | `7.04 → −0.12` dB | `7.0432 → −0.1172` | ✓ |
| EM-GS flatness in `L` | `0.191` dB | `0.1910` dB | ✓ |
| Noiseless `σ_{L+1}/σ_1` | `≈7×10⁻¹⁶` | check 6: `6.73e-16` (L=2), `7.56e-16` (L=3) | ✓ |
| Projection-off ≡ EM-GS | `max|diff| = 0` | check 1: `0.000e+00` | ✓ |
| Automated checks | thirteen | `checks_passed 13 / 13` | ✓ |
| Spectrum cliff after Cadzow | `~10⁻³` | `after_cadzow[3] = 1.237e-3` | ✓ |
| True-channel cliff | `~10⁻¹⁶` | `true_channel[3] = 5.73e-16` | ✓ |

### Family TD (`reports/trackD_partB9_analysis.json`)

| Claim | Manuscript | Recomputed | |
|---|---|---|---|
| Zero crossings `N=16/32/64` | `0.588/0.518/0.544` | `0.5876/0.518/0.5440` | ✓ |
| Magnitude residual `N=16` vs `64` | mean `0.493`, max `0.901` dB | `0.4933`, `0.9012` | ✓ |
| Crossings bracketed | 1 of 3 (`N=32`) | `{16: False, 32: True, 64: False}` | ✓ |
| Prospective error, generator 1 | `0.13` dB | `1.30 − 1.1725 = 0.127` | ✓ |
| Prospective error, generator 2 | `0.315` dB | `error_db = 0.31515` | ✓ |
| Abstention, clustered / literal | `1.4 % / 29.1 %` | `0.01449 / 0.29091` | ✓ |
| `K`-invariance, SNR ≥ 5 / pooled | `0.095 / 0.294` dB | `0.0945 / 0.2943` | ✓ |
| Falsifier miss | `0.006` dB | `0.3 − 0.2943 = 0.0057` | ✓ |

### Prediction rule (prospective, `numpy.interp` on the 8-point A2 table)

| ρ | Rule output | Committed prediction | |
|---|---|---|---|
| `0.331` | `+1.3035` | `1.30` | ✓ |
| `0.400` | `+0.6479` | `0.6478` | ✓ |
| `0.748` | `−0.1170` | `−0.12` | ✓ |

### Complexity (`results/track_b/timing.json`)

| Claim | Manuscript | Recomputed | |
|---|---|---|---|
| HS-GS fixed order vs chained EM-GS | `1.49/1.32/1.22×` | `1.492/1.322/1.216` | ✓ |
| Order-selection share of runtime | `57.7–85.5 %` | `0.5773/0.7412/0.8550` | ✓ |

---

## D. Defects found and corrected in this pass

**Four corrections. Three are numerical; one is a family mislabel.** All were
inherited from the earlier draft `paper/hsgs.tex`, whose values were carried
across without recomputation.

### D1 — CCRB crossing magnitudes did not reproduce (corrected)

Manuscript said HS-GS lies above the CCRB "by `0.32–8.85` dB (`+0.32–1.41` at
`N=8`; `+4.29–4.82` at `N=32, P=30`)". Recomputing from
`results/track_b/b3/*.npz` against `constrained_crlb.json["constrained"]["b3"]`
with the manuscript's own pooling rule gives **`0.34–9.09` dB (`+0.34–1.44` at
`N=8, P=30`; `+4.36–4.88` at `N=32, P=30`)**. The alternative mean-of-dB pooling
gives `0.31–7.13` — so neither rule reproduces the old figures, and their
provenance could not be established. Replaced with the recomputed values; the
`% SOURCE` comment now names the store and records the superseded numbers.

The counts `34/36` and `26/36` and the `6.94` dB figure were unaffected — those
reproduce exactly.

### D2 — Shrinkage coefficients are not recoverable from the retained stores (removed, and flagged)

The old draft attached shrinkage coefficients `c_EM = 0.944`, `c_HS = 0.561`,
`0.776` and a `2.62` dB crossing figure to the bias argument. **None can be
regenerated:** the `b3` archives store squared-error numerators and the
denominator, not inner products, so `c = ⟨Ĝ,G⟩/‖G‖²_F` cannot be recomputed
without rerunning the sweep — which this task forbade. The `2.62` dB figure
also does not reproduce under either pooling rule.

These are **not restated** in the master. In their place the paragraph now
carries the crossing magnitudes that *are* verifiable:

- HS-GS falls below the CCRB at exactly two points: `N=8, P=10`, SNR `−5` and
  `0` dB, by `2.35` and `0.15` dB.
- GS and EM-GS dip below only at `N=8, P=10, −5` dB, by `0.55` and `0.88` dB,
  and stay above at every point with SNR ≥ 15 dB (nearest margin `0.84` dB).

The qualitative claim (crossings occur where the shrinking update is furthest
from unbiased, at the lowest SNR and smallest aperture) survives and is now
supported by verifiable numbers. **A new limitation (§XVIII item 19) records
that the bias explanation is circumstantial rather than quantitative, and why.**

### D3 — Runtime numbers were tagged to the wrong family (corrected)

Table IV tagged the `1.49/1.32/1.22×` runtime row as family **EXT** and the
`% SOURCE` comment read "trackB_hankel_emgs timing measurement, reported in
paper/hsgs.tex". The actual store is `results/track_b/timing.json` — a
**track_b** timing run at the B3 configuration. Retagged **B3**; source comment
now names the store, the two dictionary keys, and the exact ratios. The
order-selection share was also promoted to its own table row (it was quoted in
two places in the text but absent from the summary table).

### D4 — Spectrum figure caption named the wrong family (corrected)

Fig. 5's caption said "**Family B3 generator**, single illustrative
realisation". The store is `trackB_hankel_emgs/results/diagnostic_spectrum.json`
— the **EXT** tree. Retagged **Family EXT**, with the seed, trial index and
`cadzow_iter` added, and the three `~10^x` order-of-magnitude statements
replaced by the actual stored values (`5.7×10⁻¹⁶`, `0.130`, `1.2×10⁻³`).
Table II's "Used in" column was corrected to list Fig. 5 and §XI under EXT.

---

## E. Terminology and forbidden-phrasing sweep

Every occurrence checked in context. All are in the correct, negated form.

| Term | Occurrences | Status |
|---|---|---|
| "oracle" | 4 | all negative: *"no oracle path count anywhere"*, *"No oracle rank is used anywhere"*, *"without an oracle"*, and the abstract's *"chosen from held-out pilots"*. Never applied to the unstructured-LS reference. |
| "upper bound" | 0 | absent |
| "efficient" | 2 | both explicit disclaimers: *"We therefore do not call EM-GS efficient"*, *"We do not call it efficient — it is biased, and the bound formally governs unbiased estimators"* |
| "law" | 1 | *"it is not a law, it is not universal"* |
| "universal" | 1 | same sentence, negated |
| "out-of-model" | 1 | §XV-C is titled *"Terminology: cross-generator, not out-of-model"* and states that calling them out-of-model *"would overstate what is tested"* |
| "reproduce" | 4 | 3 refer to HS-GS reproducing EM-GS bit-for-bit or the package reproducing the comparison; 1 to the prediction rule being reproducible. **No claim of reproducing Xiao et al.** |
| "TODO" / "VERIFY CITATION" / "FIXME" / "TBD" | 0 | none |

Xiao et al. `\cite{xiao2026}` appears 4 times: as prior work that unrolls EM-GS
into a trained network, as the source of the Saleh–Valenzuela reference, and
twice in `% SOURCE` comments naming the TD analysis cells. No comparison
against their reported numbers, and no reproduction claim.

---

## F. Prospective vs post-hoc

The manuscript distinguishes these at four points, and the distinction is
load-bearing for the paper's main epistemic claim.

- **Table III caption**: *"Each prediction was committed to version control
  before the corresponding measurement was run, and the relation was not
  refitted afterwards."*
- **§XV-B** states the prediction rule explicitly enough to be re-executed:
  `numpy.interp` on a fixed 8-point table, no regression, no spline, no refit.
  All three predictions reproduce to 4 decimal places (§C above).
- **Fig. 8 caption** repeats that predictions were registered before
  measurement, and the A2 relation is drawn and labelled *"fitted on the ULA
  model only"*.
- **Fig. 4 caption** (path count): *"The hypothesis was recorded before the
  experiment was run."*
- The `K`-invariance falsifier and the `P12` crossing criterion are both quoted
  from the stored pre-registration (`B1_collapse.P12_prediction`,
  `P12_falsifier`), and **both are reported as narrowly failed or partly
  extrapolated**, not quietly dropped.

Nothing in the document presents a post-hoc fit as a prediction. The one
relation that *is* a fit (the A2 curve) is labelled as such in the figure it
appears in and in the text.

---

## G. Negative results retained

§XVIII is a 19-item list. The substantive ones, all present:

1. `N=8` is a **mixed result, not a small positive one** — gain negative above
   0 dB, worst `−2.23` dB, projection inactive in `42.5 %` of trials.
   §IX-D states *"the sign of any scalar summary at `N=8` depends on the
   pooling"* and reports per-point values rather than one number.
2. Cadzow's operator is **not idempotent**; sweep count is part of the operator
   definition, so TD (`n_cz=1`) and B3 (`n_cz=4`) numbers are never compared.
3. The **magnitude collapse across `N` is imperfect** — one-signed residual,
   mean `0.493` dB, max `0.901` dB, unexplained.
4. Two of three zero crossings are **extrapolated, not bracketed**.
5. The pre-registered `K`-invariance criterion **fails**, by `0.006` dB.
6. The second prospective prediction errs by `0.315` dB, **an order of
   magnitude worse than the first**.
7. Order selection **dominates runtime** and nothing was done about it.
8. Neither bound governs any estimator compared to it (all are biased).
9. Cross-generator tests **stay inside the ULA family** — distribution shift,
   not model violation.
10. **New this pass**: the bias explanation of the CCRB crossings is
    circumstantial; the shrinkage diagnostic is not recoverable from the
    retained stores (§D2).

The `L=16` result — gain small but **significantly negative** — is retained in
§X with its confidence interval, not rounded to zero.

---

## H. Residual risks

Things a reader should know that the audit could not close.

1. **`refs.bib` verification status is inherited, not re-checked.** All 14
   entries that actually appear in the reference list carry a `[VERIFIED-PDF]`,
   `[VERIFIED-WEB]`, `[VERIFIED-WEB+USER]` or `[VERIFIED-USER]` tag in the bib
   file. The two `[UNVERIFIED]` entries — `netrapalli2015` (page numbers) and
   `tr38901` (carried from the old draft's inline bibliography) — are **not
   cited** in the master and so do not print. No network access was used to
   re-verify DOIs this session; the tags are the earlier verification pass's.
2. **The shrinkage coefficients are gone, not corrected.** Recovering them
   needs a rerun of the B3 sweep with inner products stored. That is a
   one-command change to the sweep driver, but it is a rerun.
3. **The old draft's crossing-range numbers remain unexplained.** They do not
   reproduce under either pooling rule. The most likely explanation is a third
   pooling convention or an earlier version of `constrained_crlb.json`, but
   this could not be established from what is stored.
4. **`sec:complexity` is a subsection of §VI**, not a top-level section, so
   Table IV's "Section" column points into §VI for the two runtime rows.
5. **13 pages, not 8–12.** No content was cut to reach the band; the extra page
   is the cost of keeping all nine figures, all four tables, all three
   algorithms and the full 19-item limitations list at two-column journal
   density. The paper contains no self-shortening guidance — that judgement is
   deliberately left to the reviewer, and the candidate cuts are listed in the
   handover notes rather than inside the manuscript.

### Candidate shortening targets, if the paper must be cut

Listed for the reviewer's decision; **none of these were applied.**

1. **Two of the three experiment families.** Family EXT contributes Figs. 3, 4,
   5, 6 and two Table IV rows, and duplicates the aperture result that B3
   already carries. Dropping EXT is the single largest cut available (~2 pages)
   and costs the controlled path-count experiment and the spectrum diagnostic.
2. **§VII (Exact Likelihood and Performance Bounds).** The full rank-one Fisher
   and tangent-space CCRB derivations run ~1.5 pages. They are the paper's only
   original theory, but a journal version could cite Gorman–Hero and
   Stoica–Ng and state the result.
3. **Algorithms 1–3.** ~0.75 pages of pseudocode. Algorithm 1 (EM-GS update) is
   prior work and could be a one-line reference; Algorithms 2 and 3 are the
   contribution.
4. **§XVIII Limitations, 19 items.** Compresses to a paragraph at the cost of
   the document's main value as a research record.
5. **Table II (experiment families).** Removable only if the paper is cut down
   to a single family — otherwise it is what keeps the three sets of numbers
   from being read against each other.
