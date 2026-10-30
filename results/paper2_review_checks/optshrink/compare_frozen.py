"""Read-only predictions compared on common cases; creates one derived JSON."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path('results/paper2_review_checks/optshrink')
source=root/'summary.json'
diagnostic=json.loads(source.read_text())
files={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
specs=list(diagnostic['cells'])
snr_ids=[f'snr_{s:+d}' for s in [-10,-5,0,5,10,15,20]]
seeds=[9101,9102,9103]
rng=np.random.default_rng(2026102892)
cells={};samples={};verified_error=0.
for cell in specs:
    comparison=diagnostic['cells'][cell]
    energy=np.array(comparison['energy'])
    errors=[]
    for seed in seeds:
        p=Path('results/paper2_review_checks/cells')/cell/f'adaptive10_s{seed}.npz'
        files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p) as f:
            assert np.array_equal(f['energy'],energy),'energies/case-order mismatch'
            errors.append(f['error'].copy())
    error=np.stack(errors)
    opt_error=np.array(comparison['methods']['optshrink']['error'])
    p=Path('results/paper2_review_checks/cells')/cell/'em100.npz'
    files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p) as f:
        assert np.array_equal(f['energy'],energy)
        delta=float(np.max(abs(f['error']-np.array(comparison['methods']['none']['error']))))
        verified_error=max(verified_error,delta)
        assert np.allclose(f['error'],comparison['methods']['none']['error'],rtol=1e-12,atol=1e-12)
    counts=rng.multinomial(len(energy),np.full(len(energy),1/len(energy)),size=4000)
    learned_db=10*np.log10(error.sum(1)/energy.sum())
    opt_db=float(10*np.log10(opt_error.sum()/energy.sum()))
    learned_boot=(10*np.log10((error@counts.T)/(counts@energy)[None])).mean(0)
    opt_boot=10*np.log10((counts@opt_error)/(counts@energy))
    gain=opt_boot-learned_boot
    samples[cell]=gain
    cells[cell]=dict(cases=len(energy),adaptive_seed_nmse_db=learned_db.tolist(),
        adaptive_mean_seed_nmse_db=float(learned_db.mean()),optshrink_nmse_db=opt_db,
        adaptive_gain_vs_optshrink_db=float(opt_db-learned_db.mean()),
        paired_case_interval_95=np.quantile(gain,[.025,.975]).tolist(),
        adaptive_severe_fraction=float(np.mean(error/energy>1)),
        optshrink_severe_fraction=float(np.mean(opt_error/energy>1)))
macro_sample=np.mean([samples[cell] for cell in snr_ids],axis=0)
macro=dict(adaptive_mean_seed_nmse_db=float(np.mean([cells[cell]['adaptive_mean_seed_nmse_db'] for cell in snr_ids])),
    optshrink_nmse_db=float(np.mean([cells[cell]['optshrink_nmse_db'] for cell in snr_ids])),
    adaptive_gain_vs_optshrink_db=float(np.mean([cells[cell]['adaptive_gain_vs_optshrink_db'] for cell in snr_ids])),
    paired_case_interval_95=np.quantile(macro_sample,[.025,.975]).tolist())
obj=dict(scope='Exploratory frozen adaptive10 versus independently likelihood-tuned final EM100-plus-OptShrink on identical test worlds; no retraining or test-based selection.',
    methods='Adaptive model unchanged; OptShrink rank7 follows EM100 once at the end. Different update budgets: not an equal-compute architecture ablation.',
    uncertainty='4000 paired whole-case bootstrap draws within each cell; common case weights for all three fixed learned seeds and OptShrink, independent draws across cells; unadjusted intervals conditional on fixed checkpoints and development selection.',
    input_hashes=files,analysis_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    common_em100_max_error_delta=verified_error,main_snr_macro=macro,cells=cells)
(root/'frozen_adaptive_comparison.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
print(json.dumps(obj,indent=2))
