"""Receiver-only adaptive/global routing and optional exact EM continuation.

The policy is calibrated offline. Inference uses only the supplied fitting
observations, known reference and noise variance, and evaluates one chosen
start per observation. The three fit seeds are separate estimators.
"""
import torch
from tcom_start_budget_20261020 import study as previous
from tcom_runtime_repair_20261020 import inference as latest
from tcom_pilot_em_20261018 import continuation

CROSSOVERS = (20., 25., 30., 35.)
TAILS = (0, 10, 20, 40)
TOLERANCE_DB = 1e-10


def plans(obs, policy):
    """Return observable-only routing records, including effective tail length.

    Continuous SNR controls model choice. The nearest declared reference/SNR
    bins control the tail lookup; a different model family in that lookup
    invokes the explicitly recorded zero-tail fallback.
    """
    crossover = float(policy['crossover_db'])
    if crossover not in CROSSOVERS:
        raise ValueError('crossover must be one of the prespecified values')
    if len(obs) == 0:
        raise ValueError('a nonempty observation batch is required')
    result = []
    for reference_key, snr_key, snr_db in latest.receiver_keys(obs):
        ordinary = snr_db <= 20. + TOLERANCE_DB
        if ordinary:
            family, steps, mismatch = 'adaptive', 0, False
        else:
            family = ('global' if snr_db >= 40. - TOLERANCE_DB or
                      snr_db > crossover + TOLERANCE_DB else 'adaptive')
            record = policy['tails'][reference_key][snr_key]
            if record['family'] not in ('adaptive', 'global') or record['steps'] not in TAILS:
                raise ValueError('invalid frozen tail record')
            mismatch = record['family'] != family
            steps = 0 if mismatch else int(record['steps'])
        result.append(dict(family=family, steps=steps, ordinary=ordinary,
            reference_key=reference_key, snr_key=snr_key,
            snr_db=float(snr_db), mismatch_fallback=mismatch))
    return result


@torch.no_grad()
def hybrid(obs, seed, policy, est, zero_tail=False):
    """Evaluate one frozen fit; zero_tail is a separate diagnostic arm."""
    plan = plans(obs, policy)
    groups = {}
    for index, row in enumerate(plan):
        steps = 0 if zero_tail else row['steps']
        key = (row['family'], steps, row['ordinary'], row['reference_key'], row['snr_key'])
        groups.setdefault(key, []).append(index)
    output = None
    for (family, steps, ordinary, _, _), indices in groups.items():
        whole_batch = len(indices) == len(obs)
        ids = None if whole_batch else torch.tensor(indices, dtype=torch.long, device=obs.z.device)
        part = obs if whole_batch else obs.batch(ids)
        estimate = est['starts'].start(family, part, seed=int(seed))
        if not ordinary:
            if steps:
                estimate = continuation.estimate(part, estimate, steps)
            estimate = previous.finish(part, estimate)
        if whole_batch:
            return estimate
        if output is None:
            output = torch.empty((len(obs), obs.z.shape[1], obs.s.shape[1]),
                                 dtype=estimate.dtype, device=estimate.device)
        output[ids.to(output.device)] = estimate
    return output
