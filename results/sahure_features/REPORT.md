# Feature-access check: results

Small-budget diagnostic: 4,096 development training worlds, 12 epochs, three matched seeds. No manuscript or old checkpoint changes.

Positive gain means the full controller has lower error. Selected checkpoints use only held-out likelihood.

| Bank | Full compared with | Gain (dB) | Adjusted conditional interval | Worst primary-cell gain |
|---|---|---:|---|---:|
| bank1 | spectral | 1.245 | [1.198, 1.293] | -0.262 |
| bank1 | measurement | 0.568 | [0.529, 0.610] | -0.398 |
| bank1 | context | 2.226 | [2.162, 2.292] | -0.259 |
| bank2 | spectral | 1.216 | [1.165, 1.266] | -0.248 |
| bank2 | measurement | 0.577 | [0.537, 0.617] | -0.388 |
| bank2 | context | 2.213 | [2.144, 2.282] | -0.209 |

Primary averages exclude weak-reference and two-path stress cases. All cells, final checkpoints, seed-specific scores and severe errors are in summary.json.
Removing controller inputs does not remove the underlying spectral operator, physical noise model or channel-dependent threshold scale. All models store 188 parameters; masked input weights have no effect.
Intervals resample paired test worlds, not independent training populations. Lack of a positive contrast does not establish equivalence.
These small-budget results cannot replace the broad study or prove which features are needed at full training scale. Review training curves, both banks and individual seeds before a manuscript claim.
