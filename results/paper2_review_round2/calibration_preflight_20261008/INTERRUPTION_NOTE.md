# Calibration preflight interruption

This partial run was deliberately interrupted before its results were analyzed.
The resume guard compared tuple-valued configuration fields with JSON lists,
which would reject a legitimate restart. No numerical or scientific rule was
changed. The original helper source is retained as `common_original.py`.

The complete run in the sibling `calibration/` directory uses canonicalized
JSON values in that guard. It regenerates the same predetermined cases and
recomputes the estimates. These partial files are retained for provenance, not
included as extra independent evidence or combined with the completed run.
