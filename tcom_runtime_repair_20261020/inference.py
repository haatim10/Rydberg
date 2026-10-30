"""Backend-only optimization of the frozen filter-continuation estimator.

The canonical engine and optimized engine share every decision table, model
weight, stage count, EM continuation, and final filter. Only evaluation of the
learned ten-stage start changes from small Torch operations to NumPy/SciPy.
"""
import json
import torch
from tcom_followups_20261017.common import REPO
from tcom_incumbent_guard_20261019 import inference as canonical
from .native import NativeEstimator

SNRS=canonical.SNRS
RSRS=canonical.RSRS
SEEDS=canonical.SEEDS
OLD_POLICY=canonical.OLD_POLICY
model_paths=canonical.model_paths
receiver_keys=canonical.receiver_keys
choices_for=canonical.choices_for
LATEST_POLICY_PATH=REPO/'results/tcom_filter_continuation_20261019/policy.json'
LATEST_POLICY=json.loads(LATEST_POLICY_PATH.read_text())

CanonicalEstimator=canonical.RepairEstimator


class OptimizedEstimator(CanonicalEstimator):
    def __init__(self,models=None):
        super().__init__(models)
        self.native_models={name:NativeEstimator(model) for name,model in self.models.items()}

    @torch.no_grad()
    def original(self,obs,model_name):
        return self.native_models[model_name](*obs.args())


def canonical_engine(models=None):
    return CanonicalEstimator(models)


def optimized_engine(models=None):
    return OptimizedEstimator(models)


@torch.no_grad()
def latest_estimate(obs,model_name,estimator):
    return estimator.repair(obs,model_name,LATEST_POLICY['repair_choices'])
