# Reading, verifying and reproducing the research

## Get the complete archive

```sh
git clone --branch codex https://github.com/haatim10/Rydberg.git
cd Rydberg
```

Topic branches are cumulative checkpoints. Use `archive/complete` when you need all studies,
their shared dependencies and the latest paper in one checkout.

## Environment

The numerical dependency pins are recorded in `sahure/requirements.txt` and the
older root `requirements.txt`. The original pilot README specifies Python 3.12.
Study protocols record their actual environment and precision. Preserve those
records when comparing results rather than assuming another Python/library
version or CPU will give identical runtimes.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r sahure/requirements.txt
```

Some tests also need `pytest`, as listed in the root requirements. The archive
tool below uses only the Python standard library. No dependency installation or
model training is needed to read reports or compile the supplied Overleaf ZIP.

## Verify downloaded evidence without running experiments

```sh
python3 tools/research_archive.py verify
```

This checks sizes and SHA-256 hashes for the included new research files and
the available previously tracked files recorded in the dated inventory. It does
not modify any experiment. `--local-only` additionally checks omitted files and
is appropriate only on the original complete local workspace or after restoring
those files. Do not run `inventory` over an existing dated inventory.

The archive's verification is different from an experiment's original full-data
audit. Original manifests and saved pass records remain unchanged. Some full-data
audits count every epoch checkpoint or hash training banks intentionally omitted
from this GitHub archive; those audits need the full local data restored first.

## Use the learned core

The `sahure/README.md` interface describes tensors and checkpoint loading. The
core uses magnitudes `Z`, pilots `S`, reference `B` and total complex noise
variance. Do not substitute a per-real-component variance without conversion.
Use a checkpoint matching the intended experiment and configuration.

The complete receiver has additional stored policies and imports earlier study
modules. Do not copy only its final `inference.py` file out of this repository;
use the complete checkout and the protocol's selected weights/policy files.

## Regenerate a study

Read its original README, `protocol.json` or `config.json` first. Training,
validation, selection, confirmation, accuracy and timing are different phases.
Do not choose a new model based on the final test results.

Many historical drivers use fixed output locations under `results/` or
`experiments/`. Run regeneration in a **separate checkout**, with a fresh output
location where supported. Do not overwrite the frozen records shipped here.
Creating a directory with the same name is not sufficient to recreate its seed
streams, data ordering or environment; follow the saved configuration and source.

The following original guides give the intended commands:

- [Self-supervised core/pilot](../sahure/README.md)
- [URformer published-setting attempt](../urformer_reproduction_20261022/README.txt)
- [Matched architecture/supervision experiment](../matched_supervision_20261023/README.txt)
- [Final receiver report](../results/tcom_reliability_20261021/RESULTS.txt)
- [SNR audit](../experiments/urformer_snr_audit/REPORT.md)
- [Cluster audit](../experiments/urformer_cluster_audit/REPORT.md)

Some original commands expect all intermediate checkpoints or original training
banks. This archive includes selected/final weights and regeneration sources,
not every intermediate training state. Exact byte-for-byte regeneration across
different numerical environments is not promised.

## Interpret the results correctly

Keep final-epoch and validation-selected results separate. Mean-setting NMSE in
dB and pooled linear NMSE are different summaries. Zero observed BER does not
mean zero population BER. Runtime is tied to the recorded CPU, precision,
thread count, initialization, stopping policy and included operations.

The latest merged paper and the four-user URformer comparisons use different
physical normalizations and pilot counts. Their headline numbers are not a
single matched experiment. See each section guide before reusing a table.
