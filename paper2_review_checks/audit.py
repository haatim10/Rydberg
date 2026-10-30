"""Recalculate every saved error and verify paired-world/source provenance."""
import json
from pathlib import Path
import numpy as np
from .run import ROOT,SEEDS,NCASE,specs,load_inputs,sha,write

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert all(sha(p)==h for p,h in manifest['sources'].items())
    assert all(sha(p)==h for p,h in manifest['banks'].items())
    assert all(sha(e['path'])==e['sha256'] for e in manifest['models'].values())
    frozen=json.loads(Path('results/sahure_expansion/frozen_selection.json').read_text())
    assert all(sha(p)==h for p,h in frozen['sources'].items())
    methods=['em100','hs_residual','hs_rician']+[f'{m}_s{s}' for m in ['adaptive10','prefix5'] for s in SEEDS]
    maximum_error_difference=0.;checked=0;phase_equivariance={}
    for spec in specs():
        data=load_inputs(spec)
        energy=np.sum(abs(data['truth'])**2,axis=(-2,-1))
        for method in methods:
            with np.load(ROOT/'cells'/spec['id']/f'{method}.npz') as f:
                assert f['estimates'].shape==(NCASE,32,3)
                error=np.sum(abs(f['estimates']-data['truth'])**2,axis=(-2,-1))
                np.testing.assert_array_equal(error,f['error'])
                np.testing.assert_array_equal(energy,f['energy'])
                maximum_error_difference=max(maximum_error_difference,float(np.max(abs(error-f['error']))))
                checked+=1
                if spec.get('mismatch')=='phase':
                    with np.load(ROOT/'cells'/'snr_+5'/f'{method}.npz') as base:
                        expected=base['estimates']*np.exp(1j*np.deg2rad(spec['value']))
                        discrepancy=float(np.max(abs(expected-f['estimates']))/max(1e-30,np.max(abs(expected))))
                        phase_equivariance[f'{spec["id"]}/{method}']=discrepancy
                        assert discrepancy<1e-8
    result=dict(state='passed',method_output_files_checked=checked,unique_worlds=9*NCASE,
        condition_cases=len(specs())*NCASE,maximum_saved_error_recalculation_difference=maximum_error_difference,
        checkpoint_hashes_unchanged=True,source_hashes_unchanged=True,original_expansion_frozen_sources_unchanged=True,
        bank_hashes_unchanged=True,constant_reference_phase_equivariance_relative_errors=phase_equivariance,
        caveat='These are frozen-model exploratory diagnostics, not retraining or independent confirmation.')
    write(ROOT/'verification.json',result)
    counts=np.random.default_rng(2026100829).multinomial(NCASE,np.full(NCASE,1/NCASE),size=2000)
    mismatches={}
    for spec in specs():
        if 'mismatch' not in spec:continue
        comparison={}
        for method in ['em100','hs_residual','hs_rician','adaptive10','prefix5']:
            names=[f'{method}_s{s}' for s in SEEDS] if method in ['adaptive10','prefix5'] else [method]
            base=[np.load(ROOT/'cells'/'snr_+5'/f'{name}.npz') for name in names]
            changed=[np.load(ROOT/'cells'/spec['id']/f'{name}.npz') for name in names]
            e0=np.stack([f['error'] for f in base]);e1=np.stack([f['error'] for f in changed])
            a0=np.stack([f['aligned_error'] for f in base]);a1=np.stack([f['aligned_error'] for f in changed])
            delta=float((10*np.log10(e1.sum(1)/e0.sum(1))).mean())
            bootstrap=(10*np.log10((e1@counts.T)/(e0@counts.T))).mean(0)
            comparison[method]=dict(degradation_db=delta,interval=np.quantile(bootstrap,[.025,.975]).tolist(),
                aligned_degradation_db=float((10*np.log10(a1.sum(1)/a0.sum(1))).mean()))
            for f in base+changed:f.close()
        mismatches[spec['id']]=comparison
    write(ROOT/'mismatch_summary.json',dict(definition='positive is worse than the clean estimator on identical observations and truth; paired case-bootstrap conditional on frozen checkpoints; phase-aligned errors are oracle diagnostics only',cells=mismatches))
    write(ROOT/'artifact_hashes.json',{str(p):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
