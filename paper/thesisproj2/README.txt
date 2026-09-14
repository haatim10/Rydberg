PAPER B — Training-Design Sensitivity of Structural-Prior Gains
===============================================================

Main file:  haatim_thesisproj2.tex
Class:      IEEEtran (journal), built in with Overleaf
Build:      pdflatex twice.  No bibtex -- the bibliography is a manual
            thebibliography block inside the .tex.
Result:     6 pages, zero undefined references, zero errors.

Figures (all in fig/):
  fig1_attribution.pdf        Fig. 1  where the gain over EM-GS comes from
  fig5_allcells.pdf           Fig. 2  every SNR bin, every training seed
  fig4_pilots_three_way.pdf   Fig. 3  pilot efficiency vs pilot-count
                                      generalization

Overleaf: upload this whole folder (or the zip) as a new project and set
haatim_thesisproj2.tex as the main document.


What this paper says
--------------------
A training loss averaged over a wide SNR range is dominated by its
low-noise tail -- 89.7% of the measured gradient falls below 5 dB here.
So any mechanism that helps at high SNR can look as though it supplied
information the network could not learn by itself.  The paper proposes a
matched-adequacy control, applies it to our own Hankel-structured
estimator, and reports that the structural gain moves from +1.286 dB
under the conventional loss to near zero under focused retraining and
negative under a corrected loss.

The result is the PAIRED per-seed change (-1.27 and -1.46 dB, both
intervals excluding zero).  The endpoints are not separated from zero on
three seeds and the paper says so explicitly.


Style note
----------
Written in the same plain style as Paper A: short sentences, worked
numbers, every term explained where it first appears.
