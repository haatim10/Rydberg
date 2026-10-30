# Submission notes (not part of the five-page paper)

## Matched learned baseline: still outstanding

The revision notes requested a URformer-style Table II row with NMSE,
processing time and severe-error rate on the same protocol. No such saved
evaluation was found. It would be misleading to copy earlier results.

The existing ordinary model in `results/paper2_long_training` has 1,586,900
parameters (recorded in its result files). It trained on 80,000 labelled
cases with N=32, K=3, P=20, RSR=10 dB and fixed pilots. The present method
has 188 parameters and uses 16,384 unlabelled training cases, P=30, varying
pilot realizations and a different training population. Parameter count is
for the local URformer-style implementation, not a verified count for the
published model. These counts and operating differences are now disclosed
in the paper; unmatched performance scores are not added to Table II.

Next scientific step: freeze a matched comparison protocol, evaluate all
methods on the same untouched cases, record full initialization/inference
time and severe errors, and distinguish transfer of an old checkpoint from
retraining at the present operating point. No experiment was launched for
this editorial revision. Do not call the baseline request completed.

Evidence used: the earlier longer-training results report and the saved
ordinary-model result files, including `seed1_U_13ep/result.json`.

## Companion manuscript

Reference [9] uses the title of the supplied `ThesisProj-7.pdf` and remains
labelled unpublished. No public identifier has been invented. Before
submission, make the companion method accessible through an actual public
preprint or the venue's permitted supplementary material and update the
reference accordingly.

## AI-assistance disclosure

The acknowledgment has been removed from this requested paper version, not
from the historical record. Keeping disclosure only in these notes does not
satisfy a venue requirement to disclose in the manuscript itself.

IEEE guidance, checked 24 September 2026, requires disclosure of generated
content (including text, figures and code) beyond ordinary language editing:
https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/

Before submitting to IEEE, restore an accurate disclosure in the required
location and confirm the target venue's instructions. For another venue,
follow that venue's policy. Use a consistent factual disclosure in Paper 1.

Suggested text to review before submission:

> OpenAI Codex assisted with drafting and revising all sections, mathematical
> exposition, experiment and analysis code, and preparation of the figures.
> Numerical plots were generated from saved simulation outputs. The author
> is responsible for the scientific claims, references and final content.

## Scope of this revision

This is a shorter, compile-checked source release. It does not resolve the
remaining scientific comparison or independently replay all experiments.
Keep the weak-reference and few-path losses, failed protection criterion,
supervised-comparison qualifications, numerical-fallback note and limitations.
