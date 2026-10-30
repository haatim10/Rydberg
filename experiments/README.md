# URformer reproduction audits

These are completed investigations of the gap between our URformer reproduction
and the published curves. They do not change the final receiver or manuscript.

| Audit | Main finding |
| --- | --- |
| [SNR convention](urformer_snr_audit/REPORT.md) | The proposed common noise shift is rejected; retain `reproduction_ensemble`. |
| [Clustered subrays](urformer_cluster_audit/REPORT.md) | None passes the requested global-LMMSE gate; new neural training was not triggered. |

Read [the audit guide](../docs/sections/04-audits/README.md) for what these findings
do and do not establish. Code, configurations and reports in both directories
are frozen. Large raw input banks are documented in the
[local-only inventory](../docs/archive_20261030/local_only_files.csv).
