import numpy as np
import pytest

from src import validation


def test_row_signs_and_depth_ratio():
    sounding = np.array([-100.0, -50.0])
    model_bed = np.array([-200.0, -100.0])  # twice as deep everywhere
    row = validation._row("own", "m", model_bed, sounding)
    assert row["n_cells"] == 2
    assert row["median_bias_m"] == pytest.approx(-75.0)  # negative: too deep
    assert row["rmse_m"] == pytest.approx(np.sqrt((100**2 + 50**2) / 2))
    assert row["depth_ratio"] == pytest.approx(2.0)


def test_row_bed_above_sea_level_contributes_no_depth():
    row = validation._row("own", "m", np.array([10.0, -50.0]), np.array([-50.0, -50.0]))
    assert row["depth_ratio"] == pytest.approx(0.5)


def test_common_cell_sets_only_name_known_models():
    for models in validation.COMMON_CELL_SETS:
        assert set(models) <= set(validation.RAW_BED_LOADERS)
