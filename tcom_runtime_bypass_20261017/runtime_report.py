"""Readable report of the completed, frozen runtime comparison."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'results/tcom_runtime_bypass_20261017'


def main():
    n=json.loads((ROOT/'runtime/nmse_summary.json').read_text())
    b=json.loads((ROOT/'runtime_ber/summary.json').read_text())
    a=json.loads((ROOT/'runtime_audit.json').read_text())
    assert a['passed']
    labels=[('learned_mean','Learned model, mean of three separate fits'),
        ('sparse_matched','Sparse, matched budget'),('sparse_10ms','Sparse, 10 ms budget'),
        ('sparse_20ms','Sparse, 20 ms budget'),('sparse_50ms','Sparse, 50 ms budget'),
        ('fast_full','Sparse, full 100 x 25')]
    lines=['Completed runtime-matched sparse comparison','',
        'The optimized sparse solver was validated against the original solver to numerical precision. '
        'It retains the exact Rician likelihood, spectral initialization, angle dictionary and monotone '
        'FISTA sparse fit; FFTs and cached quadratic calculations reduce implementation overhead.', '',
        'Controls were selected from 180 fresh development channel realizations using only 21 fitting '
        'and 9 held-out pilot magnitudes. Candidate families span 64/128/256 dictionary atoms, '
        '1/2/4/8/16/25 inner updates, and 2/4/8/12/20/40/80/100 outer updates. '
        'The receiver selects settings from known noise variance; it does not receive true path count or channel.', '',
        f"The matched cap was {n['caps_ms']['sparse_matched']:.3f} ms, the learned model's calibrated p90 latency. "
        'Separate fresh timing verifies every selected estimator. Timings below include initialization, '
        'observation-dependent setup and parameter dispatch, use one CPU thread and float64/complex128, '
        'and exclude loading models and generating data. They are implementation measurements, not hard deadlines.', '',
        'Method | fresh p90 ms | equal-cell mean NMSE dB | pooled linear-error NMSE dB | BER %']
    for name,label in labels:
        lines.append(f"{label} | {n['timing'][name]['p90_ms']:.3f} | {n['methods'][name]['primary_nmse_db']:.3f} | "
                     f"{n['methods'][name]['pooled_ratio_of_sums_db']:.3f} | {100*b['overall']['methods'][name]['ber']:.5f}")
    c=n['contrasts']['sparse_matched']
    gain=-c['gain_db'];lo,hi=c['familywise95_adjusted_ci_db']
    lines+=['',f"The learned model's advantage on the prespecified equal-cell dB criterion is {gain:.3f} dB "
        f"(simultaneous 95% interval {[-round(hi,3),-round(lo,3)]}). The observed sparse-20-ms aggregate "
        'channel error is similar; its difference interval includes zero, which is not a formal equivalence result.', '',
        'The conclusion depends on operating conditions and the metric. Sparse wins at 0 dB and mostly at '
        '5 dB; the learned model wins every tested 10, 15 and 20 dB channel-error cell. Averaging each '
        'condition in dB gives those conditions equal weight. Pooling squared errors instead makes the '
        'noisiest cells dominate and favors matched sparse. Aggregate BER also favors matched sparse. '
        'The report therefore does not claim overall or uniform superiority for the learned estimator.', '',
        f"Learned aggregate BER: {100*b['overall']['methods']['learned_mean']['ber']:.5f}%. "
        f"Matched sparse BER: {100*b['overall']['methods']['sparse_matched']['ber']:.5f}%. "
        f"Relative bit-error reduction from matched sparse: {100*(1-b['overall']['methods']['sparse_matched']['ber']/b['overall']['methods']['learned_mean']['ber']):.2f}%.", '',
        'Test scope: 960 new channel realizations, 64 per condition; SNR 0/5/10/15/20 dB; RSR 12 dB; '
        '2, 4 or random 3-7 paths per user; N=32, K=3, P=30. Every estimator uses the same pilot '
        'observations and the same exact magnitude-domain QPSK detector on 1,474,560 bits per estimator. '
        'Three learned fits are evaluated separately, not ensembled or counted as extra independent data.', '',
        'Bootstrap intervals preserve repeated path-count vectors by sharing case resamples across SNR '
        'within each regime: 192 resampling clusters. They condition on the trained fits and selected '
        'controls. Rare-error claims remain limited by the number of channel realizations.', '',
        'All pooled p90 caps passed. The 50 ms arm slightly exceeded 50 ms in three individual SNR '
        'strata (about 50.04-50.24 ms); these measurements were retained and controls were not reselected.', '',
        'Checks passed: original and optimized solver parity, finite-difference gradient, exact zero-penalty '
        'EM equivalence, a deliberately triggered FISTA restart, eligibility and validation winner checks, '
        'unchanged checkpoint/source hashes, raw-error recomputation and 150 single-case prediction replays.', '',
        'Paper implication: the strongest measured advantage is efficient moderate/high-SNR channel '
        'estimation. Include the low-SNR and BER results as well as the runtime curve; the primary NMSE '
        'score alone would hide an important tradeoff.', '',
        'Figures: runtime/runtime_accuracy.png; runtime/nmse_by_snr.png; runtime_ber/runtime_ber.png.',
        'Full values and uncertainty: runtime/nmse_summary.json; runtime/final_timing.json; runtime_ber/summary.json.',
        'Audit: runtime_audit.json. Protocol, frozen selections, per-world predictions and bit counts are retained.']
    (ROOT/'RESULTS_RUNTIME.txt').write_text('\n'.join(lines)+'\n')
    print(ROOT/'RESULTS_RUNTIME.txt')


if __name__=='__main__':main()
