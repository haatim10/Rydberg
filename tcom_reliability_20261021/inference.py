"""Observable residual check and intensity-regression fallback.

The incumbent is the immutable B-proposal hybrid. Only weak-reference cases
above nominal30dB can enter the new branch. Fitting likelihood acceptance is
an observable safeguard, not a channel-error or out-of-sample guarantee.
"""
import json
import numpy as np
import torch
from tcom_followups_20261017.common import REPO
from tcom_hybrid_20261020 import inference as hybrid_inference
from tcom_hybrid_20261020 import study as previous
from tcom_runtime_repair_20261020.inference import receiver_keys
from tcom_runtime_bypass_20261017.sparse_fast import spectral_initial
from tcom_pilot_em_20261018 import continuation

BASE_POLICY_PATH = REPO/'results/tcom_hybrid_sensitivity_20261020/protocol.json'
BASE_POLICY = json.loads(BASE_POLICY_PATH.read_text())['proposal_policy']
CONDITION_LIMIT = 1e8
ACCEPTANCE_MARGIN = 1e-12
RULES = tuple(dict(metric=metric,threshold=threshold,tail=tail)
              for metric in ('mean','max') for threshold in (2.,4.,8.) for tail in (10,20))


def normalize_rule(rule):
    """Validate scientific fields; identifiers are inert reporting metadata."""
    if rule is None:
        return None
    if not isinstance(rule,dict) or not all(key in rule for key in ('metric','threshold','tail')):
        raise ValueError('rule requires metric, threshold and tail')
    normalized={key:rule[key] for key in ('metric','threshold','tail')}
    if normalized not in RULES:
        raise ValueError('rule must belong to the fixed 12-member menu')
    return normalized


def _arrays(obs, *, allow_zero_variance=False):
    arrays = [np.asarray(x.detach().cpu().numpy(),dtype=np.complex128 if i in (1,2) else np.float64)
              for i,x in enumerate(obs.args())]
    z,s,b,v = arrays
    if (z.ndim!=3 or s.ndim!=3 or b.shape!=z.shape or s.shape[0]!=z.shape[0]
            or s.shape[-1]!=z.shape[-1] or v.shape!=(len(z),) or len(z)==0):
        raise ValueError('aligned nonempty batched observations required')
    if not all(np.isfinite(x).all() for x in arrays) or np.any(z<0):
        raise ValueError('finite observations and nonnegative magnitudes required')
    if np.any(v<0 if allow_zero_variance else v<=0):
        raise ValueError('invalid supplied complex noise variance')
    return arrays


def intensity_design(s):
    """Real design for y=|cS|^2+2Re(cS), with K=3 and 15 columns."""
    s = np.asarray(s,dtype=np.complex128)
    if s.ndim!=2 or s.shape[0]!=3 or not np.isfinite(s).all():
        raise ValueError('finite S[3,P] required')
    columns = [2*s.real.T,-2*s.imag.T,np.abs(s.T)**2]
    for i in range(3):
        for j in range(i+1,3):
            value = s[i]*s[j].conj()
            columns.extend((2*value.real[:,None],-2*value.imag[:,None]))
    return np.concatenate(columns,axis=1)


@torch.no_grad()
def algebraic_start(obs):
    """Return (g, valid, diagnostics), with zero placeholders when invalid.

    One SVD least-squares solve per case shares all antenna right-hand sides.
    The quadratic coefficients are nuisance variables; only the six linear
    coefficients recover c=g/b. A reference must repeat exactly over pilots.
    Zero variance is supported here solely for algebraic noiseless checks.
    """
    z,s,b,v = _arrays(obs,allow_zero_variance=True)
    batch,n,p = z.shape
    if s.shape[1]!=3:
        raise ValueError('this declared initializer requires K=3')
    g = np.zeros((batch,n,3),dtype=np.complex128)
    valid = np.zeros(batch,dtype=bool)
    rank = np.zeros(batch,dtype=np.int64)
    condition = np.full(batch,np.inf)
    reason = np.full(batch,'unprocessed',dtype='<U40')
    for i in range(batch):
        if np.any(b[i,:,0]==0) or not np.array_equal(b[i],np.repeat(b[i,:,:1],p,axis=1)):
            reason[i]='reference_not_repeated_nonzero';continue
        design = intensity_design(s[i])
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            rhs = ((z[i]**2-v[i])/np.abs(b[i])**2-1).T
        if not np.isfinite(design).all() or not np.isfinite(rhs).all():
            reason[i]='nonfinite_design_or_rhs';continue
        try:
            coef,_,rank[i],singular = np.linalg.lstsq(design,rhs,rcond=None)
        except np.linalg.LinAlgError:
            reason[i]='least_squares_failed';continue
        if len(singular)==15 and singular[-1]>0:
            condition[i]=singular[0]/singular[-1]
        if rank[i]!=15:
            reason[i]='rank_deficient';continue
        if not np.isfinite(condition[i]) or condition[i]>CONDITION_LIMIT:
            reason[i]='ill_conditioned';continue
        estimate = (coef[:3]+1j*coef[3:6]).T*b[i,:,0,None]
        if not np.isfinite(estimate).all():
            reason[i]='nonfinite_estimate';continue
        g[i]=estimate;valid[i]=True;reason[i]='valid'
    return torch.from_numpy(g),valid,dict(rank=rank,condition=condition,invalid_reason=reason,
        pilots=np.full(batch,p,dtype=np.int64),design_columns=15)


@torch.no_grad()
def finish(obs,g):
    """Apply the unchanged inherited filter, grouping mixed conditions."""
    keys = [(ref,snr) for ref,snr,_ in receiver_keys(obs)]
    if len(set(keys))==1:
        return previous.finish(obs,g)
    output = g.clone()
    for key in dict.fromkeys(keys):
        ids = torch.tensor([i for i,k in enumerate(keys) if k==key],dtype=torch.long)
        output[ids]=previous.finish(obs.batch(ids),g[ids])
    return output


@torch.no_grad()
def fallback(obs,steps):
    """Algebraic start, one selected exact EM tail, then inherited filter.

    Invalid rows remain zero placeholders and valid=False; the caller decides
    the explicit invalid branch. No spectral initializer is called here.
    """
    if int(steps)!=steps or steps not in (10,20):
        raise ValueError('fallback tail must be 10 or 20')
    _arrays(obs)
    initial,valid,diag = algebraic_start(obs)
    output = initial.clone()
    if valid.any():
        ids = torch.from_numpy(np.flatnonzero(valid))
        try:
            values = finish(obs.batch(ids),continuation.estimate(obs.batch(ids),initial[ids],steps))
            finite = torch.isfinite(values).all((-2,-1)).numpy()
            output[ids[finite]]=values[finite]
            bad = ids.numpy()[~finite]
            valid[bad]=False;diag['invalid_reason'][bad]='nonfinite_refinement'
        except (FloatingPointError,np.linalg.LinAlgError):
            valid[ids.numpy()]=False;diag['invalid_reason'][ids.numpy()]='refinement_failed'
    diag.update(valid=valid.copy(),tail=int(steps))
    return output,valid,diag


@torch.no_grad()
def algebraic20(obs,diagnostics=False):
    """Standalone baseline; invalid algebraic rows use spectral+EM20."""
    estimate,valid,diag = fallback(obs,20)
    invalid = ~valid
    if invalid.any():
        ids = torch.from_numpy(np.flatnonzero(invalid));part=obs.batch(ids)
        z,s,b,_ = _arrays(part)
        initial = torch.from_numpy(spectral_initial(z,s,b))
        estimate[ids]=finish(part,continuation.estimate(part,initial,20))
    diag['spectral_invalid_branch']=invalid.copy()
    return (estimate,diag) if diagnostics else estimate


def fitting_nll(obs,g):
    z,s,b,v = _arrays(obs)
    return continuation.nll(z,g.detach().cpu().numpy()@s+b,v).mean((-2,-1))


@torch.no_grad()
def from_base(obs,incumbent,rule,diagnostics=False,*,force_fallback=False):
    """Shared gate. Forced eligibility is for calibration/checks only."""
    rule=normalize_rule(rule)
    if force_fallback and rule is None:
        raise ValueError('forced calibration requires a valid fallback rule')
    keys = receiver_keys(obs)
    eligible = np.array([ref=='5' and snr>30.+1e-10 for ref,_,snr in keys])
    count = len(obs)
    score = np.full(count,np.nan);row_scores=np.full((count,obs.z.shape[1]),np.nan)
    trigger = np.zeros(count,dtype=bool);valid=trigger.copy();accepted=trigger.copy()
    before=np.full(count,np.nan);candidate_loss=before.copy();after=before.copy()
    condition=np.full(count,np.nan);rank=np.zeros(count,dtype=np.int64)
    reason=np.full(count,'not_triggered',dtype='<U40')
    output = incumbent
    if rule is not None and eligible.any():
        ids = torch.from_numpy(np.flatnonzero(eligible));part=obs.batch(ids)
        z,s,b,v = _arrays(part)
        residual = z-np.abs(incumbent[ids].detach().cpu().numpy()@s+b)
        rows = 2*np.mean(residual**2,axis=-1)/v[:,None]
        row_scores[ids.numpy()]=rows
        selected = rows.mean(-1) if rule['metric']=='mean' else rows.max(-1)
        score[ids.numpy()]=selected
        trigger[ids.numpy()]=True if force_fallback else selected>rule['threshold']
        if trigger.any():
            tids=torch.from_numpy(np.flatnonzero(trigger));part=obs.batch(tids)
            alternative,ok,fd = fallback(part,rule['tail'])
            valid[tids.numpy()]=ok;condition[tids.numpy()]=fd['condition'];rank[tids.numpy()]=fd['rank']
            reason[tids.numpy()]=fd['invalid_reason']
            old_loss=fitting_nll(part,incumbent[tids]);before[tids.numpy()]=old_loss;after[tids.numpy()]=old_loss
            if ok.any():
                local=torch.from_numpy(np.flatnonzero(ok));new_loss=fitting_nll(part.batch(local),alternative[local])
                candidate_loss[tids.numpy()[ok]]=new_loss
                accept = np.isfinite(new_loss)&(new_loss<old_loss[ok]-ACCEPTANCE_MARGIN)
                chosen=tids[local[accept]]
                accepted[chosen.numpy()]=True;after[chosen.numpy()]=new_loss[accept]
                if len(chosen):
                    output=incumbent.clone();output[chosen]=alternative[local[accept]]
    if diagnostics:
        return output,dict(score=score,row_scores=row_scores,eligible=eligible,trigger=trigger,valid=valid,
            accepted=accepted,incumbent_nll=before,fallback_nll=candidate_loss,selected_nll=after,
            condition=condition,rank=rank,invalid_reason=reason,metric=None if rule is None else rule['metric'],
            threshold=None if rule is None else rule['threshold'],tail=0 if rule is None else rule['tail'],
            triggered=trigger,rank_invalid=trigger&~valid,fallback_condition=condition,
            nll_before=before,nll_after=after,pilots=obs.z.shape[-1],forced_calibration=bool(force_fallback))
    return output


@torch.no_grad()
def gated(obs,seed,rule,est,base_policy=None,diagnostics=False,*,force_fallback=False):
    """Compute the complete incumbent, then at most one triggered fallback."""
    policy = BASE_POLICY if base_policy is None else base_policy
    incumbent = hybrid_inference.hybrid(obs,seed,policy,est)
    return from_base(obs,incumbent,rule,diagnostics=diagnostics,force_fallback=force_fallback)
