# Track-A reproduction audit — Cui et al., *Towards Atomic MIMO Receivers*

Final status of the Track-A reproduction. All five published figures have
been run to completion against the validated Cui implementation. No channel,
solver, calibration or metric was tuned to improve agreement.

---

## 1. Summary table

| Figure | Cui configuration | Our configuration | Qualitative match | Absolute match | Measured discrepancy | Likely reason |
|---|---|---|---|---|---|---|
| **Fig. 5** — NMSE vs SNR | 16-QAM, N×K=36×3, RSR 12 dB, SNR −5…12 | identical, 2 000 trials/SNR, 18 pts | ✅ full | ⚠️ offset | **≈ 2 dB**, ours better, roughly constant | channel conditioning: Tr((AA^H)^-1) ≈1.57× larger in Cui's implied channel. Audited twice; see `SPEC_38901_AUDIT.md` |
| **Fig. 6** — NMSE vs RSR | 16-QAM, N×K=36×3, SNR 3 dB, RSR 0…25 | identical, 500 trials/RSR, 26 pts | ✅ full | ⚠️ offset | **≈ 2 dB**, same constant offset | same cause as Fig. 5 |
| **Fig. 7(a)** — BER vs SNR | 4-QAM, N×K=36×3, RSR 12 dB | identical, 333 000 trials, 18 pts | ✅ full | ✅ **yes** | SNR shift **−0.90…+0.05 dB**; BER ratio 0.89–0.93 | within pixel-extraction uncertainty |
| **Fig. 7(b)** — BER vs SNR | 16-QAM, N×K=100×6, RSR 12 dB | identical, 122 000 trials, 18 pts | ✅ full | ✅ **yes** | SNR shift **−0.10…−0.15 dB**; BER ratio 0.95–0.97 | best agreement of all five |
| **Fig. 8** — BER vs RSR | 4-QAM (body text), N×K=36×3, SNR 3 dB | identical, 239 000 trials, 26 pts | ✅ full | ✅ **yes** | RSR shift **−0.80…−1.40 dB**; BER ratio 0.78–0.87 | see §3 caption inconsistency |

Two quantities Cui states in prose were reproduced **without any curve
extraction**, and both land inside his stated range:

| Cui's claim | Our measurement |
|---|---|
| Fig. 7(b): EM-GS↔ZF SNR gap "between 3 ∼ 4 dB" | **3.79 – 3.97 dB** (BER 3e-2 … 1e-3) |
| Fig. 8: EM-GS "more than one order of magnitude" BER reduction, RSR 0→20 dB | **39.1×** (and 49.4× for biased GS) |

---

## 2. Store integrity

Every store: single config fingerprint, zero duplicate keys, 100 % `status=ok`,
trial indices contiguous from 0, common random numbers preserved (all
algorithms evaluate the same frozen world at each operating point). BER is
always **global bit errors ÷ global bit count**, never a mean of per-trial
BERs; NMSE is always a ratio of sums, never a mean of per-trial dB.

| Store | Points | Trials | Rows | Bits/algorithm | Fingerprint |
|---|---|---|---|---|---|
| `fig5_final` | 18 SNR | 2 000/pt | — | n/a (NMSE) | `925f2ab8…` |
| `fig6` | 26 RSR | 500/pt | 52 000 | n/a (NMSE) | `925f2ab8…` |
| `fig7a` | 18 SNR | 333 000 | 3 330 000 | 1 998 000 | `87f2337c…` |
| `fig7b` | 18 SNR | 122 000 | 732 000 | 2 928 000 | `17aef19f…` |
| `fig8` | 26 RSR | 239 000 | 2 390 000 | 1 434 000 | `87f2337c…` |
| `fig8_16qam` | 5 RSR | 15 000 | 150 000 | 180 000 | `925f2ab8…` |

Shared fingerprints are **by design**: `fingerprint_payload` deliberately
excludes the sweep grids, so two figures with the same physical
configuration share an identity and are separated by `experiment` name and
directory (Fig. 5/6, and Fig. 7(a)/8). Verified non-colliding.

---

## 3. Fig. 8 — caption vs body text

The paper contradicts itself:

* **Caption:** "…for a 16-QAM modulator under 3 dB SNR."
* **Body text (§VI-C):** "The SNR is fixed as 3 dB and a 4-QAM modulator is adopted."

Both were run. Against Cui's published curve levels over 12 matched
(algorithm, RSR) points:

| | 4-QAM | 16-QAM |
|---|---|---|
| median BER ratio to Cui | **0.82** | 24.12 |
| median \|log₁₀ ratio\| | **0.086** | 1.367 |

The 16-QAM variant sits ≈ 29× above the 4-QAM curve at equal RSR and misses
Cui's plotted levels by more than an order of magnitude at every point beyond
RSR = 0.

> **The body text is correct; the caption is in error. Cui's published Fig. 8
> is a 4-QAM figure.**

---

## 4. Monte-Carlo resolution — are more trials needed?

No. Assessed per point rather than assumed.

The deepest zero-error points have one-sided 95 % Wilson upper bounds of
**1.13e-5** (240 000 bits) and **7.5e-6** (360 000 bits). Cui's Fig. 7 axes
floor at 10⁻⁴ and his curves leave the plot near 5e-5, so our bounds sit
**4–7× below anything the paper plots**. Every comparison the published
figures support is already resolved.

More trials would only be needed to extend the curves *below* Cui's own
plotted range, which serves no reproduction purpose.

---

## 5. Answers

**1. Which figures are successfully reproduced?**
Fig. 7(a), Fig. 7(b) and Fig. 8 — qualitatively *and* absolutely, within
0.1–1.4 dB and BER ratios 0.78–0.97, plus two independently reproduced
prose claims.

**2. Which are only qualitatively reproduced?**
Fig. 5 and Fig. 6. Every ordering, slope, crossing, gap and asymptote is
correct (including the exact 3.01 dB CRLB↔ZF separation and ZF flatness in
RSR), but the absolute level is offset by ≈ 2 dB.

**3. Are the remaining differences likely implementation errors?**
No. The evidence is against it:
* the offset is *constant* across both NMSE sweeps — a signature of channel
  statistics, not of a solver, SNR/RSR or metric bug;
* the BER figures, which use the *same* channel, solvers and calibration,
  agree to within 0.1–1.4 dB;
* two prose claims reproduce independently of any extraction;
* two separate audits against the actual TR 38.901 specification found no
  correction that closes the gap (`SPEC_38901_AUDIT.md`).
The residual is attributable to channel conditioning arising from
information Cui does not specify (§7).

**4. Is Track A sufficiently validated to freeze?**
**Yes.** Three of five figures reproduce absolutely; the other two reproduce
in every qualitative respect with a documented, twice-audited offset whose
cause is identified and whose magnitude is stable.

**5. What is validated for reuse in Track B?**
* exact observation model `Z = |GS + B + W|`, `W ~ CN(0, σ²)`
* biased GS (Cui Alg. 1) and EM-GS (Alg. 2) incl. Bessel ratio R(κ)=I₁/I₀
* exact-model Fisher/CRLB, ZF-known-phase, exhaustive LS and ML searches
* SNR eq. (36) and RSR eq. (37) calibration (single-user denominator)
* NMSE (ratio of sums, 10log₁₀) and bit-level BER (global errors/bits)
* Gray-mapped QAM alphabet and demapper
* CRN trial construction and the resumable/fingerprinted store

**6. Any scientific reason to run more trials before freezing?**
No — see §4.

**7. What remains unreproducible?**
* **CM-ZF** — described only as extending ref. [39] to the biased PR problem.
  Not a specification; deliberately not invented.
* **Absolute NMSE level in Figs. 5/6** — requires the array geometry and the
  orientation of μ_eg relative to it. Cui gives only "N = 36". A sweep showed
  the achievable shift spans 0.0–1.3 dB purely by rotating an unspecified
  frame, so the residual cannot be resolved from published information.
* **Fig. 8's intended modulation** — resolved empirically here (§3), but the
  paper itself is self-contradictory.
