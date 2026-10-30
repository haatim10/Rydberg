"""Read-only audit and independent checkpoint replay of the small learning study."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import paper2_baseline_learning as study


def db(values):
    return float(10*np.log10(np.mean(values)))


def audit():
    torch.set_num_threads(1)
    root = study.OUT
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['source_hashes'] == study.provenance()
    assert json.loads((root/'status.json').read_text())['state'] == 'complete'
    report = {'source_hashes_verified':len(manifest['source_hashes']), 'runs':[], 'data_hashes_verified':0}
    for name,spec in study.STUDIES.items():
        data = {}
        for split in ('train','val'):
            path = root/name/'data'/f'{split}.pt'
            meta = json.loads(path.with_suffix('.json').read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']
            assert meta['spec'] == spec
            data[split] = study.Tensors(torch.load(path,weights_only=True))
            assert len(data[split]) == spec[split]
            report['data_hashes_verified'] += 1
        assert not set(data['train'].arrays['trial'].tolist()) & set(data['val'].arrays['trial'].tolist())
        records = []
        summary = json.loads((root/name/'summary.json').read_text())
        for arm in study.ARMS:
            folder = root/name/arm
            r = json.loads((folder/'result.json').read_text()); records.append(r)
            f = np.load(folder/'validation_errors.npz')
            errors = f['errors']
            assert errors.shape == (spec['epochs'],spec['val'])
            assert np.isfinite(errors).all() and (errors>=0).all()
            np.testing.assert_array_equal(f['trial'],data['val'].arrays['trial'])
            np.testing.assert_array_equal(f['snr_db'],data['val'].arrays['snr_db'])
            curve = np.array([db(row) for row in errors])
            np.testing.assert_allclose(curve,[h['validation_nmse_db'] for h in r['history']],rtol=0,atol=1e-12)
            chosen,selection = study.select_epoch(curve,errors[curve.argmin()],'one_se')
            assert chosen+1 == r['chosen_epoch'] and selection == r['selection']
            assert len(list(folder.glob('epoch_*.pt'))) == spec['epochs']
            model,_ = study.build(arm)
            assert study.weight_hash(model) == r['initial_hash']
            for ep in set((chosen+1,spec['epochs'])):
                state = torch.load(folder/f'epoch_{ep:03d}.pt',weights_only=True)
                model.load_state_dict(state['model'])
                np.testing.assert_allclose(study.evaluate(model,data['val']),errors[ep-1],rtol=0,atol=1e-12)
            state = torch.load(folder/'checkpoint.pt',weights_only=False)
            model.load_state_dict(state['model'])
            assert abs(db(study.evaluate(model,data['train']))-r['final_training_nmse_db'])<1e-12
            assert all(np.isfinite(h['max_gradient_norm']) for h in r['history'])
            report['runs'].append({'study':name,'arm':arm,'epochs':spec['epochs'],
                'selected_epoch':chosen+1,'final_training_db':r['final_training_nmse_db'],
                'final_validation_db':curve[-1], 'checkpoint_replay':'exact',
                'initial_db':r['initial_nmse_db'],
                'max_gradient_norm':max(h['max_gradient_norm'] for h in r['history'])})
        assert records[0]['initial_hash'] == records[1]['initial_hash']
        assert [h['data_order_sha256'] for h in records[0]['history']] == [h['data_order_sha256'] for h in records[1]['history']]
        for label in ('final','selected'):
            arrays = []
            for arm,r in zip(study.ARMS,records):
                arr = np.load(root/name/arm/'validation_errors.npz')['errors']
                arrays.append(arr[-1 if label=='final' else r['chosen_epoch']-1])
            for group,values in summary['scores'][label].items():
                if group == 'all':
                    mask = np.ones(len(data['val']),dtype=bool)
                else:
                    low,high = map(float,group[1:-1].split(','))
                    snr = data['val'].arrays['snr_db'].numpy()
                    mask = (snr>=low)&(snr<high)
                assert values['n'] == mask.sum()
                assert abs(values['normalization_gain_db']-(db(arrays[0][mask])-db(arrays[1][mask])))<1e-12
        for steps in (10,100):
            saved = np.load(root/name/f'emgs_{steps}_validation.npz')
            arr = saved['errors']
            np.testing.assert_array_equal(saved['trial'],data['val'].arrays['trial'])
            np.testing.assert_array_equal(saved['snr_db'],data['val'].arrays['snr_db'])
            np.testing.assert_allclose(arr,study.evaluate(study.Classical(steps),data['val']),rtol=0,atol=1e-12)
            for group,value in summary['classical_validation_nmse_db'][str(steps)].items():
                if group == 'all':
                    mask = np.ones(len(arr),dtype=bool)
                else:
                    low,high = map(float,group[1:-1].split(','))
                    mask = (saved['snr_db']>=low)&(saved['snr_db']<high)
                assert abs(db(arr[mask])-value)<1e-12
    report['matched_initial_weights_and_order'] = True
    report['validation_only'] = True
    report['previous_frozen_source_entries_verified'] = {}
    for name in ('paper2_original_channel','paper2_reproduction','paper2_reproduction_diagnostic'):
        old = json.loads((study.ROOT/'results'/name/'manifest.json').read_text())
        for path,digest in old['source_hashes'].items():
            assert hashlib.sha256((study.ROOT/path).read_bytes()).hexdigest() == digest
        report['previous_frozen_source_entries_verified'][name] = len(old['source_hashes'])
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    audit()
