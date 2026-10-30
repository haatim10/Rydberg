# Research archive — 30 October 2026

This is the user-selected research archive, not a full backup of the local
workspace. Original experiment and manuscript files retain their content and
paths. No new training, numerical experiment or paper revision was performed
for this upload.

## Included and local-only files

The snapshot starts from commit `0162cf515bf34354fcf151b9adbdbe7cde887711`.
It inventories new files visible under the pre-existing ignore rules. Files
already ignored by older studies remain governed by those rules; the inventory
does not claim to cover every historical cache or every file on the computer.

The upload adds **9,115 research files**, totaling **2.006 GB** before Git compression, plus the new navigation and archive documentation. **59,874 files (44.546 GB)** in the inventory stay local. Existing committed files remain in Git history.

| Record | Contents |
| --- | --- |
| [included_files.csv](included_files.csv) | Exact new research files, byte sizes, SHA-256 hashes, sections and inclusion reasons |
| [local_only_files.csv](local_only_files.csv) | Omitted files, sizes, reasons and hashes for scientific artifacts |
| [existing_files.csv](existing_files.csv) | Existing tracked files present in the sparse local checkout, hashed before organization |
| [inventory_summary.json](inventory_summary.json) | Machine-readable totals, scope and topic branches |
| [publication_verification.json](publication_verification.json) | Archive checks recorded before publication |

Temporary extractions, caches and build/process artifacts have no hash in the
local-only table. All other inventoried omitted artifacts have a SHA-256 hash.
Nothing was deleted. Third-party reference PDFs/full-text copies stay local;
see the audit README for links to the authors' original publications.

## Branches and sections

Branches are cumulative checkpoints. Each adds one topic to the preceding work,
so shared imports and dependencies remain available. Use `codex` for all topics.
Existing `main` and older branches are not rewritten. Earlier topic checkpoints
do not contain later studies, so use the full `codex` archive for full-inventory
verification and links to subsequent sections.

| Topic | New files | Size before compression | Branch |
| --- | ---: | ---: | --- |
| foundation | 6,374 | 1332.4 MB | `research/self-supervised-20261030` |
| receiver | 916 | 135.6 MB | `research/tcom-receiver-20261030` |
| comparison | 206 | 417.0 MB | `research/urformer-comparison-20261030` |
| audits | 605 | 18.3 MB | `research/urformer-audits-20261030` |
| paper | 1,014 | 102.3 MB | `paper/merged-tcom-20261030` |

## Weight selection

Files named `best`, `selected`, `final` and constant-control weights are retained.
For older epoch-numbered runs, the last saved epoch and the epoch explicitly
selected in `result.json` are retained. The frozen Hankel confirmation selection
also contributes its chosen epochs. Original filenames and bytes are unchanged;
optimizer state inside a selected checkpoint is not stripped or rewritten.
A last saved epoch is labelled as such, not asserted to be a completed final run.
Intermediate epoch files and resume-only states stay local.

## Large data and regeneration

Raw training/test banks, bulk prediction arrays, most raw numeric arrays and
large duplicate ZIPs remain local. Compact audit covariance/rank/per-cell arrays
and small `val_per_trial.npy` selection evidence are retained where listed.
Reports, protocols, seeds, histories, figures and selection records remain
available even when their raw arrays are omitted.

Use the original campaign code and protocol in a separate checkout to generate
new data. Exact byte identity across changed environments is not guaranteed.
Some historical full-data audit commands need omitted banks or every epoch;
restore those files before using those commands. Their original saved reports
and pass records are preserved without claiming that they were rerun here.

## Verify this archive

```sh
python3 tools/research_archive.py verify
```

On the original complete workspace, `--local-only` also checks the omitted files.
The verification tool does not run models or modify evidence. The included-files
manifest describes research artifacts; navigation and archive bookkeeping files
are versioned by Git rather than recursively hashed into their own manifest.
