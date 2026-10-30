# Self-supervised adaptive Hankel pilot

This is a separate experiment. It does not edit Paper 1 estimators or the
existing Paper 2 experiments. The bounded pilot is the first decision point
in the approved plan; the larger multi-condition campaign is not launched
automatically.

## Reproduce

Run from the repository root with Python 3.12 and the dependencies in
`requirements.txt` (inside this directory):

The ZIP is organized as an overlay for a Git checkout of the Rydberg
repository. The campaign runner records `git rev-parse HEAD`, so run new
campaigns inside that checkout. Regenerating reports, auditing saved results
and using saved estimators do not require Git metadata.

```sh
python -m unittest sahure.test_numerics sahure.test_pipeline -v
python -m sahure.run --phase train --output results/sahure_pilot
python -m sahure.run --phase evaluate --output results/sahure_pilot
python -m sahure.report --output results/sahure_pilot
python -m sahure.audit --output results/sahure_pilot
python -m sahure.package --output results/sahure_pilot
```

On the current workspace, the interpreter is `../../.venv-paper2/bin/python`.
The report command only regenerates statistics and figures; it does not train
or select models. The package command builds a checked ZIP with this module,
simulator dependencies, data, checkpoints, results, figures and hashes.
`--phase all` combines training and evaluation. Use a fresh
output directory for an independent full run. Existing output directories
must retain their exact configuration and frozen checkpoints.

Single-thread CPU settings are applied before numerical library imports.
The four-hour limit applies to the pilot runner's cumulative compute phases.
Checkpoints are saved after each complete epoch. Epochs are round-robin over
the six arms. If the cap occurs mid-epoch, partial weights are saved separately
and resume starts from the preceding complete epoch. Incomplete arms are not
silently included as completed comparisons. Per-method test rows are flushed
as they finish, so evaluation can resume without duplicating completed rows.

## Interface

```python
from sahure.model import AdaptiveHankelEstimator, ModelConfig
from sahure.train import load_model

model = AdaptiveHankelEstimator(ModelConfig(core="em", structure="adaptive"))
# Tensors have shapes Z/B=(batch,N,P), S=(batch,K,P), sigma2=(batch,).
# Match the model's real/complex precision; CPU training defaults to float32.
G_hat = model.estimate(Z, S, B, sigma2)
G_hat, stages = model.estimate(Z, S, B, sigma2, diagnostics=True)
# estimate also accepts one unbatched NumPy/tensor world (N,P)/(K,P),
# with scalar sigma2, and returns an (N,K) torch tensor.

model, checkpoint = load_model("results/sahure_pilot/checkpoints/em_adaptive_best.pt")
# load_model defaults to double-precision inference.
```

All estimator inputs are observations or known quantities. `Observations`
contains exactly `z`, `s`, `b`, and `sigma2`; training has no channel field.
Ground-truth files can be deleted after generation without preventing
training or validation. Model state must be loaded only from trusted local
checkpoints.

## File map

| File | Purpose |
|---|---|
| `math.py` | Stable likelihood, posterior field, Wirtinger derivative, initializer, Hankel operators |
| `model.py` | EM and GD cores, none/global/adaptive structures |
| `data.py` | Existing simulator adapter; separate observation and truth storage |
| `train.py` | Label-free masks, held-out loss, checkpoint selection and timing preflight |
| `classical.py` | Matched-objective and linearized comparisons |
| `evaluate.py` | Freeze decisions, generate fresh paired test worlds, time all estimators |
| `report.py` | Paired bootstrap, report and five pilot figure groups |
| `run.py` | Fixed pilot configuration, source hashes and compute cap |
| `test_numerics.py`, `test_pipeline.py` | Mathematical and pipeline regression checks |
| `MATHEMATICS.md` | Derivations and numerical conventions |
| `LITERATURE.md` | IEEE overlap matrix and claim boundaries |
| `NEXT_STAGE.md` | Conditional larger campaign and required future ablations |

## Recorded comparison details

The six learned arms have ten stages, 20 training epochs, and shared random
training masks/orders. Best checkpoints are selected by validation held-out
Rician NLL. The eight estimator inputs are exactly those in the approved
plan; residual and noise-power features use `log1p` for scale compression.
The global and no-structure arms do not require observation-dependent
structural decisions. An additional intervention replaces each adaptive
network's outputs with validation means, retaining its learned steps.

HS-GS uses the existing exhaustive per-world rank search with 25 candidate
updates, four Cadzow sweeps and 100 final updates. The extra hard-Hankel
Rician-GD control chooses one global rank on validation held-out likelihood,
using 25 candidate updates and one sweep, then 100 final updates. That
global choice is not a true-rank or oracle comparator. The convergence
references cap at 1,000 updates and expose the stopping status. No globally
optimal phase-retrieval solution is claimed.

Timing uses batch one, CPU single-thread and double precision for every
method. It includes initialization and per-instance selection. Learned
models use PyTorch and classical methods NumPy/SciPy, so speed claims remain
specific to these implementations. Training cost is reported separately.

## Deliberately deferred

The pilot does not run the 16,384-case mixed-condition campaign, three seeds,
SNR/RSR sweeps, alternative channels, supervised training, separately trained
final-only architecture, noise-mismatch or transfer studies. The outputs
explicitly identify those gaps. The pilot's five figure groups visualize
the evidence it actually contains, not unrun SNR curves or transfer results.
