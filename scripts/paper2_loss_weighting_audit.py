"""Read-only result audit; no training or model-selection changes."""
import json
import numpy as np
import torch
import paper2_loss_weighting as s


def audit():
    root=s.OUT
    assert json.loads((root/'status.json').read_text())['state']=='complete'
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['source_hashes']==s.sources()
    assert manifest['prior_artifact_hashes']==s.verify_old()
    summary=json.loads((root/'summary.json').read_text())
    w=json.loads((root/'weights.json').read_text())
    for split in ('train','val'):
        path=root/'development/data'/f'{split}.pt'
        meta=json.loads(path.with_suffix('.json').read_text())
        assert s.base.engine.hash_file(path)==meta['sha256']
    train=torch.load(root/'development/data/train.pt',weights_only=True)
    assert float(s.weights(train['snr_db']).mean())==w['normalizer']
    assert int((train['snr_db']>=15).sum())==w['high_snr_training_count']
    records=[]; evaluations=[]
    for arm in s.ARMS:
        folder=root/'development'/arm
        record=json.loads((folder/'result.json').read_text()); records.append(record)
        arr=np.load(folder/'validation_errors.npz'); evaluations.append(arr['errors'])
        checkpoint=torch.load(folder/'checkpoint.pt',weights_only=False)
        assert checkpoint['epoch']==11 and len(record['history'])==12
        assert len(list(folder.glob('epoch_*.pt')))==12
        np.testing.assert_array_equal(arr['errors'],checkpoint['validations'])
        curve=10*np.log10(arr['errors'].mean(axis=1))
        np.testing.assert_allclose(curve,[h['validation_nmse_db'] for h in record['history']],atol=1e-12,rtol=0)
        chosen,selection=s.base.select_epoch(curve,arr['errors'][curve.argmin()],'one_se')
        assert chosen+1==record['chosen_epoch'] and selection==record['selection']
        assert np.isfinite(arr['errors']).all()
        assert all(np.isfinite(h['max_gradient_norm']) for h in record['history'])
        for label in ('final','selected'):
            score=s.prior.scores(arr['errors'][-1 if label=='final' else chosen],arr['snr_db'])
            assert score==summary['scores'][label][arm]
    assert records[0]['initial_hash']==records[1]['initial_hash']
    assert [h['data_order_sha256'] for h in records[0]['history']]==[h['data_order_sha256'] for h in records[1]['history']]
    np.testing.assert_array_equal(evaluations[0],np.load(s.OLD/'development/classical_start/validation_errors.npz')['errors'])
    for label in ('final','selected'):
        scores=summary['scores'][label]; screen=summary['screen'][label]
        weighted=scores['high_snr_weighted']
        assert abs(scores['ordinary_reference']['all']['nmse_db']-weighted['all']['nmse_db']-screen['overall_gain_vs_ordinary_db'])<1e-12
        assert abs(weighted['[15,20)']['nmse_db']-summary['references']['validation']['10']['[15,20)']['nmse_db']-screen['high_snr_deterioration_vs_initial_db'])<1e-12
    result={'source_entries_verified':len(manifest['source_hashes']),
            'data_hashes_verified':2,'runs':2,'epochs':24,'exact_previous_control_replay':True,
            'matched_initial_weights_and_order':True,'selection_and_scores_recomputed':True,
            'driver_checkpoint_replays_exact':summary['checkpoint_replays_exact'],
            'weights':w,'screen':summary['screen']}
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    audit()
