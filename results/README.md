# Saved experimental results

Each subdirectory is a separate campaign. Its name usually matches a module at
the repository root, or a driver in `scripts/`. Use the
[experiment catalog](../docs/EXPERIMENT_CATALOG.md) to find code and reports.

| File type | Meaning |
| --- | --- |
| `RESULTS.txt`, `REPORT.md`, `RESULTS_REPORT.md` | Human-readable findings and limits |
| `summary.json`, `nmse_summary.json` | Numerical results; preserve the stated aggregation |
| `protocol.json`, `config.json` | Settings, seed streams, selection rules and assumptions |
| `policy.json`, `selection*.json` | Frozen receiver choices or model selection |
| `checks*.json`, `audit*.json`, `*verification*.json` | Saved checks; read their scope and date |
| `completion.json`, `status.json` | Historical execution state, not a live job monitor |
| `best.pt`, `selected.pt`, `final.pt`, selected `epoch_*.pt` | Model weights with different selection meanings |
| PNG, SVG and PDF figures | Visualizations of the associated recorded study |

Large raw banks and intermediate weights remain local under the user's selected
archive policy. [The inventory](../docs/archive_20261030/README.md) identifies the
included and excluded files individually. No excluded file was deleted.

Start with [the final receiver](tcom_reliability_20261021/RESULTS.txt),
[matched URformer comparison](matched_supervision_20261023/RESULTS.txt), and
[URformer reproduction attempt](urformer_reproduction_20261022/RESULTS.txt).
The later audits live under [`experiments/`](../experiments/README.md).
