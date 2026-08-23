# `formulae.tex` — complete formula reference

Every equation, convention and constant used in Track A and Track B, with the
source file each was transcribed from.

## Build

```
pdflatex formulae.tex
pdflatex formulae.tex      # twice, for the ToC and cross-references
```

Requires only standard packages: `amsmath`, `amssymb`, `mathtools`, `booktabs`,
`longtable`, `enumitem`, `xcolor`, `hyperref`, `microtype`, `fancyhdr`,
`caption`, `geometry`, `lmodern`.

## Status

Structurally validated (environments, braces, `$` parity, cross-references,
text-mode underscores) but **not compiled** — no TeX distribution is installed
in the container it was written in. Compile once before relying on the layout.

## Contents

| Section | Covers |
|---|---|
| 1 | Notation; conventions C1–C5 (amplitude not intensity, 10log10, ratio-of-sums, noise inside the magnitude, CRN scope) |
| 2 | Observation model, canonical form, the conjugation verified three ways |
| 3 | Both channel models: 38.901 clustered, geometric ULA |
| 4 | SNR and RSR calibration, both factor-K traps, measured values with CIs |
| 5 | QAM normalisation and Gray mapping |
| 6 | Spectral init (incl. magnitude rescale and phase anchor), biased GS, EM-GS, genie ZF, exhaustive LS/ML |
| 7 | Rician CRLB, the 3.0103 dB phase-loss gap, the unconstrained caveat |
| 8 | Hankel operator, Kronecker's theorem, the relaxation, rank cap, Cadzow, HS-GS, order selection, redundancy, mode-modulus |
| 9 | NMSE, BER, Wilson, paired bootstrap, gain/win/tie/trimmed/median, adaptive trial rule, complexity |
| 10 | Every figure in both tracks with the equations that generated it |
| 11 | What the formulae do and do not establish |

57 numbered equations, 67 labels, 0 undefined references.
