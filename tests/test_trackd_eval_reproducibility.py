"""PROMPT 15 A2 -- evaluation on a given checkpoint must be reproducible.

Three properties, each of which was in doubt before PROMPT 15 A1 and each of
which is now pinned:

1. ``stage4.evaluate`` is deterministic: two calls in one process on the same
   checkpoints give bitwise identical per-trial NMSE.
2. The stored rows in ``reports/trackD_stage4_results.json`` are reproducible
   from the committed ``best.pt`` -- so the results file and the checkpoints
   on disk agree, and no published number is orphaned from its weights.
3. Any helper that builds an evaluation dataset uses the SPECTRAL initialiser.
   This is the one that actually failed. ``TrackDConfig().train.init``
   defaults to ``"random"`` and the stage scripts all override it;
   ``scratch/p12_partC_eval.py`` did not, so PROMPT 12 Part C evaluated
   spectrally-trained networks on a random ``G0``.

Property 3 is the load-bearing one and it is cheap. Properties 1 and 2 run a
handful of trials, which is enough to catch a weight or world mismatch: the
observed error when the initialiser was wrong was 4.8x relative on trial one,
not a rounding difference.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

S4 = Path("results/track_d/stage4")
RESULTS = Path("reports/trackD_stage4_results.json")
N_TRIALS = 3

pytestmark = pytest.mark.skipif(
    not (S4 / "C_U1_snr5_20" / "best.pt").exists() or not RESULTS.exists(),
    reason="stage-4 checkpoints or results not present in this checkout",
)


def _cfg():
    from trackD_urformer.config import TrackDConfig
    cfg = TrackDConfig()
    # Exactly what stage4.main() does before it evaluates.
    return replace(cfg, train=replace(cfg.train, init="spectral"))


def test_default_train_init_is_random_so_evaluators_must_override():
    """The trap itself, pinned. If this default ever changes to 'spectral'
    the override becomes redundant rather than wrong, and this test says so."""
    from trackD_urformer.config import TrackDConfig
    assert TrackDConfig().train.init == "random", (
        "config default changed; re-read every evaluation helper before "
        "assuming the spectral override is still needed"
    )


def test_partC_eval_uses_spectral_init():
    """scratch/p12_partC_eval.eval_models must not inherit the 'random'
    default. Checked by the G0 it hands the model, not by reading source."""
    from scratch import p12_partC_eval as pc
    from trackD_urformer.baselines import make_initial_G
    from trackD_urformer.dataset import TrackDDataset

    captured = {}
    real = TrackDDataset.g0

    def spy(self, idx):
        captured["init"] = self.init
        return real(self, idx)

    TrackDDataset.g0 = spy
    try:
        pc.eval_models({}, (5.0, 20.0), n_test=1)
    finally:
        TrackDDataset.g0 = real

    assert captured.get("init") == "spectral", (
        f"eval_models built its dataset with init={captured.get('init')!r}; "
        "the arms are trained with the spectral initialiser"
    )
    assert make_initial_G is not None  # import is part of the contract


def test_evaluate_is_deterministic_in_process():
    from trackD_urformer.stage4 import HIGH_SNR, PART_C, evaluate

    cfg = _cfg()
    a = evaluate(cfg, PART_C, HIGH_SNR, N_TRIALS, False)
    b = evaluate(cfg, PART_C, HIGH_SNR, N_TRIALS, False)
    for k in PART_C:
        np.testing.assert_array_equal(
            np.asarray(a["per_trial_nmse"][k]),
            np.asarray(b["per_trial_nmse"][k]),
            err_msg=f"{k}: evaluate() is not deterministic in-process",
        )
    np.testing.assert_array_equal(np.asarray(a["snr_db"]),
                                  np.asarray(b["snr_db"]))


def test_stored_stage4_rows_reproduce_from_committed_checkpoints():
    """The published part_c rows must come back from best.pt, bitwise."""
    from trackD_urformer.stage4 import HIGH_SNR, PART_C, evaluate

    stored = json.loads(RESULTS.read_text())["part_c"]
    got = evaluate(_cfg(), PART_C, HIGH_SNR, N_TRIALS, False)

    np.testing.assert_array_equal(
        np.asarray(stored["snr_db"][:N_TRIALS], dtype=float),
        np.asarray(got["snr_db"], dtype=float),
        err_msg="test worlds differ from the ones the stored rows used",
    )
    for k in PART_C:
        np.testing.assert_array_equal(
            np.asarray(stored["per_trial_nmse"][k][:N_TRIALS], dtype=float),
            np.asarray(got["per_trial_nmse"][k], dtype=float),
            err_msg=(f"{k}: stored rows do not reproduce from best.pt -- the "
                     "results file and the checkpoint have drifted apart"),
        )


def test_partC_training_config_sets_spectral_init():
    """The PROMPT 12 Part C training driver must not inherit the 'random'
    default either. This is the defect that invalidated those runs: the arms
    were TRAINED from a random initial estimate while the stage-4 arms they
    were compared against were trained from the spectral one, so the contrast
    varied the initialiser as well as the seed."""
    from dataclasses import replace  # noqa: F401  (used by run_cfg)
    from scratch.p12_partC import run_cfg
    from trackD_urformer.config import TrackDConfig

    spec = {"snr": (5.0, 20.0), "seed": 2, "hankel": False}
    assert run_cfg(TrackDConfig(), spec).train.init == "spectral"


@pytest.mark.parametrize("rel,expected", [
    ("results/track_d/stage4/C_U1_snr5_20/best.pt", "spectral"),
    ("results/track_d/stage4/C_H1_snr5_20/best.pt", "spectral"),
])
def test_checkpoints_record_the_initialiser_they_were_trained_with(rel, expected):
    """Checkpoints carry their training config; assert the stage-4 arms are the
    spectral ones, so a future comparison against them can be checked rather
    than assumed."""
    f = Path(rel)
    if not f.exists():
        pytest.skip(f"{rel} not in this checkout")
    cfg = torch.load(f, map_location="cpu", weights_only=False)["config"]
    assert cfg["train"]["init"] == expected
