"""Angle-grid sparse EM, using only the referenced-magnitude Rician model.

The M step minimizes .5 ||A X S - (Q-B)||_F^2 + lambda ||X||_1.
The corresponding observed objective is v/2 sum RicianNLL + lambda ||X||_1.
Unit-column A is an oversampled DFT frame: A A^H=(D/N)I. A complex-valued
monotone FISTA inner solve gives a generalized EM update. Neither true G,
true angles, nor path counts are available to this estimator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from sahure.data import Observations, split_observations
from sahure.math import nll, posterior_field, spectral_initial, right_solve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/tcom_followups_20261017/sparse'
FACTORS = (0., .03, .1, .3, 1., 3.)
BUDGETS = (25, 50, 100)
SNRS = (0., 5., 10., 15., 20.)
REGIMES = (2, 4, None)
DEV_COUNT = 12
TEST_COUNT = 64


def dictionary(n, d, *, device=None, dtype=torch.float64):
    if d < n:
        raise ValueError('the tight-frame step requires D >= N')
    freq = -torch.pi + 2 * torch.pi * torch.arange(d, device=device, dtype=dtype) / d
    ant = torch.arange(n, device=device, dtype=dtype)
    return torch.exp(-1j * ant[:, None] * freq[None, :]) / np.sqrt(n)


def soft(x, threshold):
    return x * (1 - threshold / x.abs().clamp_min(1e-30)).clamp_min(0)


def quadratic(x, a, s, target, lam):
    return .5 * (a @ x @ s - target).abs().square().sum((-2, -1)) + lam * x.abs().sum((-2, -1))


@torch.no_grad()
def mstep(x, a, obs, target, lam, *, inner=25):
    gram = obs.s @ obs.s.mH
    rhs = target @ obs.s.mH
    lip = (a.shape[1] / a.shape[0]) * torch.linalg.eigvalsh(gram)[:, -1]
    step = 1 / lip[:, None, None]
    y = x.clone()
    momentum = torch.ones(len(x), dtype=x.real.dtype, device=x.device)
    objective = quadratic(x, a, obs.s, target, lam)
    initial = objective.clone()
    restarts = torch.zeros(len(x), dtype=torch.int64, device=x.device)
    for _ in range(inner):
        gradient = a.mH @ ((a @ y) @ gram - rhs)
        new = soft(y - step * gradient, step * lam[:, None, None])
        value = quadratic(new, a, obs.s, target, lam)
        bad = value > objective + 1e-10 * (1 + objective.abs())
        if bad.any():
            # Restart only the members whose accelerated proposal increased Q.
            gradient0 = a.mH @ ((a @ x[bad]) @ gram[bad] - rhs[bad])
            new[bad] = soft(x[bad] - step[bad] * gradient0,
                            step[bad] * lam[bad, None, None])
            value[bad] = quadratic(new[bad], a, obs.s[bad], target[bad], lam[bad])
            momentum[bad] = 1
            restarts += bad
        if (value > objective + 1e-8 * (1 + objective.abs())).any():
            raise FloatingPointError('proximal M step increased its objective')
        next_momentum = .5 * (1 + torch.sqrt(1 + 4 * momentum.square()))
        y = new + ((momentum - 1) / next_momentum)[:, None, None] * (new - x)
        x, momentum, objective = new, next_momentum, value
    gradient = a.mH @ ((a @ x) @ gram - rhs)
    mapped = (x - soft(x - step * gradient, step * lam[:, None, None])) / step
    relative_residual = torch.linalg.vector_norm(mapped.flatten(1), dim=1) / (
        1 + torch.linalg.vector_norm((a.mH @ rhs).flatten(1), dim=1))
    return x, dict(restarts=restarts, surrogate_decrease=initial-objective,
                   relative_proximal_residual=relative_residual)


@torch.no_grad()
def sparse_trajectory(obs, factor, *, d=128, inner=25, budgets=BUDGETS, audit=False):
    n = obs.z.shape[1]
    a = dictionary(n, d, device=obs.z.device, dtype=obs.z.dtype)
    g = spectral_initial(obs.z, obs.s, obs.b)
    x = (n / d) * (a.mH @ g)
    lam = factor * torch.sqrt(obs.sigma2 * obs.z.shape[-1] * np.log(d))
    saved = {}
    history = []
    proximal = []
    restarts = 0
    previous = .5 * obs.sigma2 * nll(obs.z, g @ obs.s + obs.b, obs.sigma2).sum((-2, -1)) + lam * x.abs().sum((-2, -1))
    max_increase = 0.
    if factor == 0:
        op = right_solve(obs.s.mH, obs.s @ obs.s.mH)
    for iteration in range(1, max(budgets) + 1):
        q = posterior_field(obs.z, g @ obs.s + obs.b, obs.sigma2)
        if factor == 0:
            # Full-row-rank A spans every channel; exact unpenalized EM.
            g = (q - obs.b) @ op
            x = (n / d) * (a.mH @ g)
            relative = torch.zeros(len(obs), dtype=obs.z.dtype)
        else:
            x, info = mstep(x, a, obs, q - obs.b, lam, inner=inner)
            g = a @ x
            restarts += int(info['restarts'].sum())
            relative = info['relative_proximal_residual']
        value = .5 * obs.sigma2 * nll(obs.z, g @ obs.s + obs.b, obs.sigma2).sum((-2, -1)) + lam * x.abs().sum((-2, -1))
        increase = float((value - previous).max())
        max_increase = max(max_increase, increase)
        if (value > previous + 1e-8 * (1 + previous.abs())).any():
            raise FloatingPointError('generalized EM increased observed penalized likelihood')
        previous = value
        if audit:
            history.append(value.cpu().tolist())
            proximal.append(relative.cpu().tolist())
        if iteration in budgets:
            saved[iteration] = g.clone()
    return saved, dict(max_observed_objective_increase=max_increase,
                      inner_restarts=restarts, final_relative_proximal_residual=relative.cpu().tolist(),
                      observed_objective_history=history, proximal_residual_history=proximal)


def estimate(obs, config):
    t = int(config['iterations'])
    return sparse_trajectory(obs, float(config['factor']), d=int(config.get('dictionary_size', 128)),
                             inner=int(config.get('inner_steps', 25)), budgets=(t,))[0][t]


def combine(items):
    return Observations(*(torch.cat([item.args()[j] for item in items]) for j in range(4)))


def sanity_checks():
    from .common import world_batch
    obs, _, _ = world_batch(2026101701, range(2), snr=5., L=2)
    a = dictionary(32, 128)
    tight_error = float((a @ a.mH - 4 * torch.eye(32)).abs().max())
    rng = torch.Generator().manual_seed(40101)
    x = torch.randn(2, 128, 3, dtype=torch.complex128, generator=rng) * .05
    direction = torch.randn(x.shape, dtype=torch.complex128, generator=rng)
    target = torch.randn(2, 32, 30, dtype=torch.complex128, generator=rng)
    def smooth_value(xx):
        return .5 * (a @ xx @ obs.s - target).abs().square().sum()
    grad = a.mH @ ((a @ x @ obs.s - target) @ obs.s.mH)
    expected = float((grad.conj() * direction).real.sum())
    eps = 1e-6
    numerical = float((smooth_value(x+eps*direction)-smooth_value(x-eps*direction))/(2*eps))
    relative_error = abs(numerical-expected) / max(1., abs(expected))
    path, info = sparse_trajectory(obs, .3, budgets=(25,), audit=True)
    em, _ = sparse_trajectory(obs, 0., budgets=(25,))
    from fair_optshrink.estimator import trajectory
    reference = trajectory(obs, [dict(rank=0,strength=0.)]*len(obs), budgets=(25,))[25]
    em_error = float((em[25]-reference).abs().max())
    if tight_error > 1e-11 or relative_error > 1e-6 or em_error > 1e-11:
        raise AssertionError('sparse estimator numerical sanity check failed')
    return dict(tight_frame_max_error=tight_error, finite_difference_relative_error=relative_error,
                zero_penalty_exact_EM_max_error=em_error, generalized_EM=info)


def protocol():
    from .common import sha, frozen_inputs
    return dict(name='Angle-domain sparse referenced-magnitude Rician EM',
                development_seed=2026101701, test_seed=2026101712,
                development_worlds_per_cell=DEV_COUNT, test_worlds_per_cell=TEST_COUNT,
                SNR_dB=SNRS, RSR_dB=12, paths=REGIMES, N=32,K=3,P=30,
                candidate_factors=FACTORS, candidate_outer_steps=BUDGETS,
                dictionary_size=128, inner_steps=25,
                penalty='lambda = factor * sqrt(sigma2 * number_of_fitting_pilots * log(128))',
                objective='sigma2/2 * sum(Rician NLL) + lambda * sum(abs(X)); G=A X',
                selection='Per observed normalized variance and reference only, mean 21/9 heldout NLL balanced across all three path regimes; no true G or L input to selection. Ties choose lower iterations then smaller factor.',
                context_masks='Random fixed masks, split seed 2026101801 + SNR integer.',
                inference='Refit selected controls on all 30 pilots.',
                independent_cells='Common generator uses local trial offsets 0 for random paths, 100000 for L2, 200000 for L4. RNG additionally keys SNR/RSR; all methods receive identical observations.',
                comparison='Three separately trained learned models (reported separately and mean scores, no ensemble), EM100, existing frozen offline OptShrink.',
                limitations=['finite off-grid dictionary', 'only one observed reference strength',
                             'finite generalized EM inner/outer budgets; global optimum not guaranteed',
                             'small independent development bank', 'single new test bank'],
                source_sha256=sha(Path(__file__)), frozen_inputs=frozen_inputs())


def run_development():
    from .common import world_batch, write_json
    OUT.mkdir(parents=True,exist_ok=True)
    spec=protocol()
    target=OUT/'protocol.json'
    if target.exists() and json.loads(target.read_text()) != json.loads(json.dumps(spec)):
        raise RuntimeError('Existing protocol differs; do not overwrite a frozen study')
    if not target.exists(): write_json(target,spec)
    write_json(OUT/'sanity_checks.json',sanity_checks())
    selections={}; all_scores={}; all_diagnostics={}
    for snr in SNRS:
        obs=combine([world_batch(2026101701, range(DEV_COUNT), snr=snr,L=L)[0] for L in REGIMES])
        context,heldout=split_observations(obs,2026101801+int(snr))
        scores=[]; diagnostics=[]
        started=time.perf_counter()
        for factor in FACTORS:
            path,info=sparse_trajectory(context,factor)
            diagnostics.append(dict(factor=factor,**info))
            for iterations in BUDGETS:
                score=nll(heldout.z,path[iterations]@heldout.s+heldout.b,heldout.sigma2).mean((-2,-1))
                scores.append(dict(factor=factor,iterations=iterations,dictionary_size=128,inner_steps=25,
                                   mean_heldout_nll=float(score.mean()),
                                   per_world_heldout_nll=score.tolist()))
        best=min(scores,key=lambda x:(x['mean_heldout_nll'],x['iterations'],x['factor']))
        selections[str(int(snr))]={k:best[k] for k in ('factor','iterations','dictionary_size','inner_steps')}
        all_scores[str(int(snr))]=scores;all_diagnostics[str(int(snr))]=diagnostics
        print(f'development SNR{snr:g}: {selections[str(int(snr))]}, score {best["mean_heldout_nll"]:.6f}, {time.perf_counter()-started:.1f}s',flush=True)
        write_json(OUT/'development_progress.json',dict(selections=selections,scores=all_scores,diagnostics=all_diagnostics))
    write_json(OUT/'frozen_selection.json',dict(protocol=spec,selections=selections,scores=all_scores,diagnostics=all_diagnostics))


@torch.no_grad()
def run_tests():
    from .common import world_batch, load_estimators, write_json, sha
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    if frozen['protocol']['source_sha256'] != sha(Path(__file__)):
        raise RuntimeError('source changed after selection')
    estimators=load_estimators()
    rows=[]
    for snr in SNRS:
        config=frozen['selections'][str(int(snr))]
        for L in REGIMES:
            label='random3-7' if L is None else f'L{L}'
            file=OUT/f'test_snr{int(snr)}_{label}.npz'
            if file.exists():
                print(f'resuming: retained {file.name}',flush=True);continue
            obs,truth,paths=world_batch(2026101712,range(TEST_COUNT),snr=snr,L=L)
            energy=np.sum(np.abs(truth)**2,axis=(1,2))
            data=dict(truth_energy=energy,paths=paths,trial_ids=np.arange(TEST_COUNT),truth_g=truth,
                      master_seed=np.asarray(2026101712),snr_db=np.asarray(snr),rsr_db=np.asarray(12.),
                      path_regime=np.asarray(label),
                      z=obs.z.numpy(),s=obs.s.numpy(),b=obs.b.numpy(),sigma2=obs.sigma2.numpy())
            started=time.perf_counter()
            for name,fn in [('sparse',lambda x:estimate(x,config)),*estimators.items()]:
                begin=time.perf_counter()
                g=fn(obs).detach().numpy()
                data[name+'_g']=g
                data[name+'_error']=np.sum(np.abs(g-truth)**2,axis=(1,2))
                data[name+'_batch_seconds']=np.asarray(time.perf_counter()-begin)
                if not np.isfinite(data[name+'_error']).all():raise FloatingPointError(name)
            np.savez_compressed(file,**data)
            print(f'test SNR{snr:g} {label}: '+', '.join(f'{name} {10*np.log10(data[name+"_error"].sum()/energy.sum()):.2f}dB' for name in ('sparse','learned_9101','em100','optshrink'))+f', {time.perf_counter()-started:.1f}s',flush=True)
    summary=analyze()
    write_json(OUT/'summary.json',summary)


def analyze():
    rows=[]
    all_trials={}
    for snr in SNRS:
        for L in REGIMES:
            label='random3-7' if L is None else f'L{L}'
            data=np.load(OUT/f'test_snr{int(snr)}_{label}.npz')
            energy=data['truth_energy']
            for key in data.files:
                if not key.endswith('_error'):continue
                name=key[:-6];error=data[key]
                rows.append(dict(snr_db=snr,path_regime=label,method=name,cases=len(error),
                    nmse_db=10*np.log10(error.sum()/energy.sum()),
                    fraction_nmse_gt_1=float(np.mean(error>energy)),
                    fraction_nmse_gt_half=float(np.mean(error>.5*energy)),
                    provisional_batch_seconds=float(data[name+'_batch_seconds'])))
                all_trials.setdefault(name,[]).append((error,energy))
    aggregate={}
    for name,values in all_trials.items():
        group=[r for r in rows if r['method']==name]
        error=np.concatenate([e for e,v in values]);energy=np.concatenate([v for e,v in values])
        aggregate[name]=dict(equal_cell_mean_nmse_db=float(np.mean([r['nmse_db'] for r in group])),
            pooled_nmse_db=float(10*np.log10(error.sum()/energy.sum())),
            fraction_nmse_gt_1=float(np.mean(error>energy)),cases=len(error))
    # Seed-averaged scores refer to separately fitted models, not an ensemble channel.
    names=[f'learned_{seed}' for seed in (9101,9102,9103)]
    aggregate['learned_seed_mean']=dict(equal_cell_mean_nmse_db=float(np.mean([aggregate[n]['equal_cell_mean_nmse_db'] for n in names])),
        fraction_nmse_gt_1=float(np.mean([aggregate[n]['fraction_nmse_gt_1'] for n in names])),
        cases=TEST_COUNT*len(SNRS)*len(REGIMES),seeds=3)
    return dict(cells=rows,aggregate=aggregate,timing_note='Batch throughput was measured during concurrent experiments and is not a latency claim. Separate idle, warmed batch-one timing is required.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('sanity','development','test','all'))
    args=parser.parse_args();torch.set_num_threads(1)
    if args.stage=='sanity':print(json.dumps(sanity_checks(),indent=2));return
    if args.stage in ('development','all'):run_development()
    if args.stage in ('test','all'):run_tests()


if __name__=='__main__':main()
