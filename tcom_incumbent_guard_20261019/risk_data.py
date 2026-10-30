"""Thirty fitting pilots plus independently drawn validation-only pilots.

The old world(P=30) supplies the channel, fitting observations and deterministic
LOS reference. Separate seed domains supply additional unconditioned CN(0,I)
pilots and CN(0,1) noise; no joint rank test or column reshuffle couples the two
sets. Truth is accessed only within make_anchor to simulate observations.
"""
from __future__ import annotations
import numpy as np
import torch
from sahure.data import Observations
from sahure_expansion.data import world
from tcom_followups_20261017.common import effective_trial_id

SNRS=(0.,5.,10.,15.,20.,22.5,25.,27.5,30.,32.5,35.,37.5,40.,42.5,45.,50.,60.)
CONTEXT_P=30
DEFAULT_HOLDOUT_SEED=2026101945
STREAM_DOMAIN=0x5249534B


def _rng(seed,index,rsr,stream,context_seed):
    bits=int(np.float64(rsr).view(np.uint64))
    return np.random.default_rng(np.random.SeedSequence([
        int(seed),STREAM_DOMAIN,int(context_seed),int(index),bits & 0xffffffff,bits >> 32,int(stream)]))


def _columns(rng,rows,columns):
    # Interleaved components preserve existing columns if the heldout count grows.
    raw=rng.standard_normal((columns,rows,2))
    return ((raw[:,:,0]+1j*raw[:,:,1])/np.sqrt(2.)).T


def make_anchor(seed,ids,L,rsr,*,snrs=SNRS,holdout_seed=DEFAULT_HOLDOUT_SEED,
                heldout_count=9,truth=False,N=32,K=3,anchor_snr=10.):
    """Return an in-memory observable anchor; never writes an artifact.

    ``truth=False`` exports no channel, field, noise realization or path counts.
    ``truth=True`` is evaluation-only and additionally exports g/unit_noise/paths.
    The first30 fitting columns are unaffected by holdout_seed/heldout_count.
    z_by_snr is retained in either mode so observation access never needs truth.
    """
    ids=list(map(int,ids));snrs=tuple(map(float,snrs));rsr=float(rsr)
    if not ids or len(ids)!=len(set(ids)) or min(ids)<0:
        raise ValueError('nonempty distinct nonnegative local indices required')
    if int(heldout_count)!=heldout_count or heldout_count<1:
        raise ValueError('positive integer heldout_count required')
    if not snrs or len(snrs)!=len(set(snrs)) or not np.isfinite([*snrs,rsr,anchor_snr]).all():
        raise ValueError('finite unique SNRs and finite reference/anchor levels required')
    heldout_count=int(heldout_count)
    arrays={key:[] for key in ('s','b','variance','z','g','unit_noise','paths')}
    indices=[]
    for index in ids:
        effective=effective_trial_id(index,L);indices.append(effective)
        w,paths=world(effective,int(seed),N=N,K=K,P=CONTEXT_P,
                      snr=float(anchor_snr),rsr=rsr,L=L)
        if not np.array_equal(w.B,np.repeat(w.B[:,:1],CONTEXT_P,axis=1)):
            raise ValueError('this helper requires the declared repeated-column LOS reference')
        hs=_columns(_rng(holdout_seed,effective,rsr,1,seed),K,heldout_count)
        hn=_columns(_rng(holdout_seed,effective,rsr,2,seed),N,heldout_count)
        hb=np.repeat(w.B[:,:1],heldout_count,axis=1)
        s=np.concatenate((w.S,hs),axis=1);b=np.concatenate((w.B,hb),axis=1)
        context_field=w.G@w.S+w.B;heldout_field=w.G@hs+hb
        zs=[]
        for snr in snrs:
            scale=10.**((float(anchor_snr)-snr)/20.)
            context_z=np.abs(context_field+w.W*scale)
            heldout_z=np.abs(heldout_field+hn*np.sqrt(w.sigma2)*scale)
            zs.append(np.concatenate((context_z,heldout_z),axis=1))
        arrays['s'].append(s);arrays['b'].append(b);arrays['variance'].append(w.sigma2);arrays['z'].append(zs)
        if truth:
            arrays['g'].append(w.G);arrays['paths'].append(paths)
            arrays['unit_noise'].append(np.concatenate((w.W/np.sqrt(w.sigma2),hn),axis=1))
    result=dict(s=np.stack(arrays['s']),b=np.stack(arrays['b']),anchor_variance=np.array(arrays['variance']),
        snrs=np.array(snrs),z_by_snr=np.stack(arrays['z']).swapaxes(0,1),
        indices=np.array(indices),local_indices=np.array(ids),seed=np.array(seed),rsr=np.array(rsr),
        holdout_seed=np.array(holdout_seed),context_P=np.array(CONTEXT_P),heldout_P=np.array(heldout_count),
        anchor_snr_db=np.array(anchor_snr),holdout_stream_domain=np.array(STREAM_DOMAIN))
    if truth:result.update(g=np.stack(arrays['g']),unit_noise=np.stack(arrays['unit_noise']),paths=np.array(arrays['paths']))
    else:assert not {'g','G','mu','truth_g','unit_noise','paths'}.intersection(result)
    return result


def observed(anchor,snr,variance_offset_db=0.,P=CONTEXT_P):
    """Read physical observations unchanged; optionally perturb supplied variance."""
    if not np.isfinite([snr,variance_offset_db]).all():raise ValueError('finite SNR/offset required')
    if int(P)!=P or not 1<=P<=anchor['s'].shape[-1]:raise ValueError('invalid requested pilot count')
    indices=np.flatnonzero(np.isclose(anchor['snrs'],snr,rtol=0,atol=1e-12))
    if len(indices)!=1:raise ValueError('requested SNR is not in the saved observable cube')
    actual=anchor['anchor_variance']*10.**((float(anchor['anchor_snr_db'])-float(snr))/10.)
    return Observations(torch.tensor(anchor['z_by_snr'][indices[0],...,:P],dtype=torch.float64),
        torch.tensor(anchor['s'][...,:P],dtype=torch.complex128),torch.tensor(anchor['b'][...,:P],dtype=torch.complex128),
        torch.tensor(actual*10.**(float(variance_offset_db)/10.),dtype=torch.float64))


def fit_holdout(obs):
    """Fixed first30/remaining split; never shuffle independent validation pilots."""
    if obs.z.shape[-1]<=CONTEXT_P:raise ValueError('combined context/heldout observations required')
    context=Observations(obs.z[...,:CONTEXT_P],obs.s[...,:CONTEXT_P],obs.b[...,:CONTEXT_P],obs.sigma2)
    heldout=Observations(obs.z[...,CONTEXT_P:],obs.s[...,CONTEXT_P:],obs.b[...,CONTEXT_P:],obs.sigma2)
    return context,heldout


def checks():
    """Simulation-boundary tests; no campaign development/test files are read."""
    from tcom_followups_20261017.common import REPO,sha,write_json
    context_checks=observable_checks=0
    for L in (2,4,None):
        for ref in (5.,12.,20.):
            kwargs=dict(snrs=(0.,10.,37.5,60.),holdout_seed=20261019450)
            public=make_anchor(20261019400,[0,1],L,ref,**kwargs)
            truth=make_anchor(20261019400,[0,1],L,ref,truth=True,**kwargs)
            assert not {'g','G','mu','truth_g','unit_noise','paths'}.intersection(public)
            for key,value in public.items():np.testing.assert_array_equal(value,truth[key])
            for i,index in enumerate((0,1)):
                w,_=world(effective_trial_id(index,L),20261019400,N=32,K=3,P=30,snr=10.,rsr=ref,L=L)
                np.testing.assert_array_equal(public['s'][i,:,:30],w.S)
                np.testing.assert_array_equal(public['b'][i,:,:30],w.B)
                np.testing.assert_array_equal(observed(public,10.).z[i].numpy(),w.Z)
                context_checks+=1
            combined=observed(public,37.5,P=39);context,heldout=fit_holdout(combined)
            assert context.z.shape==(2,32,30) and heldout.z.shape==(2,32,9)
            np.testing.assert_array_equal(context.z.numpy(),public['z_by_snr'][2,:,:,:30])
            np.testing.assert_array_equal(heldout.s.numpy(),public['s'][:,:,30:])
            changed=make_anchor(20261019400,[0,1],L,ref,snrs=kwargs['snrs'],holdout_seed=20261019451)
            for key in ('s','b'):
                np.testing.assert_array_equal(public[key][:,:,:30],changed[key][:,:,:30])
            np.testing.assert_array_equal(public['z_by_snr'][:,:,:,:30],changed['z_by_snr'][:,:,:,:30])
            assert not np.array_equal(public['s'][:,:,30:],changed['s'][:,:,30:])
            more=make_anchor(20261019400,[0,1],L,ref,heldout_count=12,**kwargs)
            np.testing.assert_array_equal(public['s'],more['s'][:,:,:39])
            np.testing.assert_array_equal(public['z_by_snr'],more['z_by_snr'][:,:,:,:39])
            reverse=make_anchor(20261019400,[1,0],L,ref,**kwargs)
            np.testing.assert_array_equal(public['s'],reverse['s'][::-1])
            np.testing.assert_array_equal(public['z_by_snr'],reverse['z_by_snr'][:,::-1])
            for offset in (-3.,3.):
                altered=observed(public,37.5,variance_offset_db=offset,P=39)
                for left,right in zip(combined.args()[:3],altered.args()[:3]):assert torch.equal(left,right)
                torch.testing.assert_close(altered.sigma2,combined.sigma2*10**(offset/10),atol=0,rtol=0)
            observable_checks+=1
    # Independent stream keys include the data seed, so future data splits never
    # accidentally share heldout noise even if the holdout master seed is reused.
    stream_a=_columns(_rng(20261019450,0,12.,1,20261019400),3,131072)
    stream_b=_columns(_rng(20261019450,0,12.,2,20261019400),3,131072)
    stream_c=_columns(_rng(20261019450,0,12.,1,20261019401),3,12)
    assert not np.array_equal(stream_a[:,:12],stream_c)
    covariance=stream_a@stream_a.conj().T/stream_a.shape[1]
    pseudocov=stream_a@stream_a.T/stream_a.shape[1]
    cross=stream_a@stream_b.conj().T/stream_a.shape[1]
    diagnostics=dict(maximum_mean=float(abs(stream_a.mean(1)).max()),
        maximum_covariance_error=float(abs(covariance-np.eye(3)).max()),
        maximum_pseudocovariance=float(abs(pseudocov).max()),maximum_cross_stream_covariance=float(abs(cross).max()))
    assert max(diagnostics.values())<.02
    result=dict(passed=True,observable_only_regime_reference_checks=observable_checks,
        exact_existing_world30_context_checks=context_checks,
        heldout_seed_and_count_do_not_change_context=True,heldout_prefix_and_index_order_reproducible=True,
        separate_pilot_noise_and_data_split_seed_domains=True,no_heldout_rank_rejection=True,
        fixed_first30_last9_split=True,supplied_variance_only_perturbations=True,stream_moment_diagnostics=diagnostics,
        source_sha256=sha(__file__),scope='Generator smoke tests only; no policy fitting or campaign test evaluation.')
    write_json(REPO/'results/tcom_incumbent_guard_20261019/risk_data_checks.json',result)
    return result


if __name__=='__main__':
    import json
    torch.set_num_threads(1)
    print(json.dumps(checks(),indent=2))
