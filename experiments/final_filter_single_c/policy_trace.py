"""Trace, for every setting, what the frozen receiver does up to the final filter.

Uses the receiver's own routing code (hybrid plans, filter_config, the
reliability gate's eligibility rule) on observables only: nominal SNR from the
supplied variance and RSR from |B|. Writes results/policy_trace.json.
"""
import json
from pathlib import Path
import numpy as np
import torch
from sahure.data import Observations
from tcom_hybrid_20261020 import inference as hybrid_inference
from tcom_reliability_20261021.inference import BASE_POLICY
from tcom_start_budget_20261020 import study as finish_module
from tcom_runtime_repair_20261020.inference import receiver_keys

HERE = Path(__file__).resolve().parent
GRID_SNRS = (0., 5., 10., 15., 20., 22.5, 25., 27.5, 30., 32.5, 35., 37.5, 40., 42.5, 45., 50., 60.)
OFF_SNRS = (31.25, 33.75, 36.25, 38.75, 47.5)
RSRS = (5., 12., 20.)
OFF_RSRS = (0., 8., 16.)


def probe(snr, rsr):
    """A one-world observation carrying only the routing observables."""
    k, n, p = 3, 32, 30
    sigma2 = k/10**(snr/10)
    b = np.full((1, n, p), np.sqrt(10**(rsr/10)), dtype=np.complex128)
    return Observations(torch.ones(1, n, p, dtype=torch.float64), torch.ones(1, k, p, dtype=torch.complex128),
                        torch.from_numpy(b), torch.tensor([sigma2], dtype=torch.float64))


def trace(snr, rsr):
    obs = probe(snr, rsr)
    plan = hybrid_inference.plans(obs, BASE_POLICY)[0]
    ref_key, snr_key, nominal = receiver_keys(obs)[0]
    cfg = finish_module.filter_config(float(ref_key), float(snr_key))
    filtered = (not plan['ordinary']) and cfg['c'] > 0 and cfg['beta'] > 0
    eligible = ref_key == '5' and nominal > 30. + 1e-10
    if plan['ordinary']:
        upstream = 'adaptive 10-stage output; no EM tail; final filter not called'
    else:
        tail = f"+ {plan['steps']} exact EM steps" if plan['steps'] else '+ 0 EM steps'
        upstream = f"{plan['family']} 10-stage output {tail}"
        if plan['mismatch_fallback']:
            upstream += ' (lookup family mismatch: zero-tail fallback)'
    if eligible:
        upstream += '; recovery check may replace it with algebraic start + 20 EM (accepted by fitting NLL) before the filter'
    return dict(snr_db=snr, rsr_db=rsr, reference_key=ref_key, snr_key=snr_key, nominal_snr_db=nominal,
                family=plan['family'], em_tail_steps=plan['steps'], ordinary=plan['ordinary'],
                mismatch_fallback=plan['mismatch_fallback'], high_snr_route=plan['family'] == 'global',
                table_choice=BASE_POLICY and finish_module.latest.LATEST_POLICY['repair_choices'][ref_key][snr_key]['choice'],
                table_c=cfg['c'] if filtered else 0., table_beta=cfg['beta'] if filtered else 0.,
                recovery_eligible=eligible, final_filter_input=upstream)


def main():
    rows = [trace(s, r) for r in RSRS + OFF_RSRS for s in GRID_SNRS + OFF_SNRS]
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/policy_trace.json').write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        if r['rsr_db'] in RSRS and r['snr_db'] in GRID_SNRS and r['snr_db'] >= 20:
            print(f"RSR {r['rsr_db']:4g} SNR {r['snr_db']:5g}: {r['family']:8s} tail {r['em_tail_steps']:2d} "
                  f"table={r['table_choice']:16s} c={r['table_c']:g} recovery={r['recovery_eligible']}")
    for r in rows:
        if r['rsr_db'] in OFF_RSRS or r['snr_db'] in OFF_SNRS:
            if r['snr_db'] in OFF_SNRS or r['snr_db'] in (30., 32.5, 35.):
                print(f"[off] RSR {r['rsr_db']:4g}->{r['reference_key']:>2s} SNR {r['snr_db']:5g}->{r['snr_key']:>4s}: "
                      f"{r['family']:8s} tail {r['em_tail_steps']:2d} mismatch={r['mismatch_fallback']} c={r['table_c']:g} recovery={r['recovery_eligible']}")


if __name__ == '__main__':
    main()
