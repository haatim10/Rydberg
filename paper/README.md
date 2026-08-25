# HS-GS manuscript

Compact IEEE-format paper on Hankel-structured GS/EM-GS channel estimation for
Rydberg atomic MIMO receivers. 6 two-column pages including references.

    pdflatex hsgs.tex && pdflatex hsgs.tex      # needs IEEEtran, algorithmicx, booktabs

## Regenerating everything

    python ../scripts/plot_paper.py     # fig/fig1..fig4, from results/track_b/**.npz
    python ../scripts/verify_paper.py   # re-derives every number in hsgs.tex

`verify_paper.py` recomputes 55 numerical claims from the per-trial stores, the
model audit and the CRLB JSON, and checks each appears verbatim in `hsgs.tex`.
It exits non-zero on any disagreement. Run it after editing either the text or
the data.

## Figures

| Figure | Claim it supports | Source |
|---|---|---|
| 1 | EM-GS attains the corrected unconstrained CRLB; HS-GS sits between the bounds | `results/track_b/b3`, `constrained_crlb.json` |
| 2 | The gain scales with array size; at N=8 the per-point gains straddle zero | `results/track_b/b3` |
| 3 | The mechanism: gain decays to zero as L reaches the Hankel rank cap | `results/track_b/b7` |
| 4 | Why the prior exists, and that Cadzow restores the cliff only approximately | seeded single realisation |

## Scope

Every result is conditional on the sparse geometric multipath model of Section
II: i.i.d. uniform AoAs, equal per-path power, flat and narrowband. Nothing here
establishes behaviour under clustered (3GPP TR 38.901 / Saleh-Valenzuela)
propagation; Section VIII states this and names the falsification test.
