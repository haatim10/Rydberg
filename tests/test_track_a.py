"""Track-A Fig. 4 numerical checks and isolation from Track B."""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim.gs import bessel_ratio
from rydberg_sim.track_a import FIG4_R10_PAPER, reproduce_fig4, track_a_fig5_spec


def test_fig4_bessel_ratio_values(tmp_path) -> None:
    values = reproduce_fig4(tmp_path / "fig4")
    assert values["R0"] == 0.0
    assert values["monotone_increasing"] is True
    assert values["bounded_01"] is True
    assert values["R10"] == pytest.approx(FIG4_R10_PAPER, abs=5e-4)
    r10 = float(np.asarray(bessel_ratio(10.0)))
    assert r10 == pytest.approx(0.9486, abs=5e-4)
    pytest.importorskip("matplotlib")
    assert (tmp_path / "fig4" / "fig4_bessel_ratio.png").is_file()


def test_fig5_spec_is_cui_not_ula() -> None:
    spec = track_a_fig5_spec(n_trials=2, snr_db_grid=(-5.0, 12.0))
    assert spec.track == "A"
    assert spec.channel_model == "cui_38901"
    assert spec.cfg.N == 36
    assert spec.cfg.K == 3
    assert spec.qam_M == 16
    assert spec.max_iter == 50
    assert spec.rsr_db_grid == (12.0,)
    assert "cm_zf" not in spec.algorithms
    assert spec.write_ber is False
