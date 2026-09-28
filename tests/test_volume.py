"""Unit tests for `volume.py`'s area-weighted submarine depth sampling."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio.transform
import shapely

from src import bed, volume


def _transform() -> rasterio.Affine:
    """1 m pixels, top-left at the origin, covering a 2x2 grid."""
    return rasterio.transform.from_origin(0.0, 2.0, 1.0, 1.0)


def test_depth_is_clipped_at_sea_level():
    """A positive bed contributes zero submarine depth; real depth is -bed."""
    elevation = np.array([[-100.0, 50.0], [0.0, -20.0]], dtype=np.float32)
    source_code = np.full((2, 2), bed.SOURCE_INVERSION, dtype=np.float32)

    stat = volume._sample_polygon(
        elevation, source_code, _transform(), shapely.box(0, 0, 2, 2)
    )

    assert stat["n_cells"] == 4
    assert stat["n_valid"] == 4
    assert stat["mean_depth_m"] == pytest.approx((100.0 + 0.0 + 0.0 + 20.0) / 4.0)
    # 1 m^2 overlap per cell, so the volume is the sum of the depths.
    assert stat["volume_change_m3"] == pytest.approx(120.0)


def test_cells_without_a_value_are_excluded_and_counted():
    elevation = np.array([[-100.0, np.nan], [np.nan, -20.0]], dtype=np.float32)
    source_code = np.full((2, 2), bed.SOURCE_BATHYMETRY, dtype=np.float32)

    stat = volume._sample_polygon(
        elevation, source_code, _transform(), shapely.box(0, 0, 2, 2)
    )

    assert stat["n_cells"] == 4
    assert stat["n_valid"] == 2
    assert stat["mean_depth_m"] == pytest.approx(60.0)


def test_polygon_with_no_value_reports_nan():
    elevation = np.full((2, 2), np.nan, dtype=np.float32)
    source_code = np.full((2, 2), np.nan, dtype=np.float32)

    stat = volume._sample_polygon(
        elevation, source_code, _transform(), shapely.box(0, 0, 2, 2)
    )

    assert stat["n_valid"] == 0
    assert np.isnan(stat["mean_depth_m"])
    assert np.isnan(stat["volume_change_m3"])


def test_interpolated_share_counts_fill_cells_only():
    elevation = np.full((2, 2), -100.0, dtype=np.float32)
    source_code = np.array(
        [
            [bed.SOURCE_INVERSION, bed.SOURCE_INTERPOLATION],
            [bed.SOURCE_INTERPOLATION, bed.SOURCE_DEM],
        ],
        dtype=np.float32,
    )

    stat = volume._sample_polygon(
        elevation, source_code, _transform(), shapely.box(0, 0, 2, 2)
    )

    assert stat["interpolated_share"] == pytest.approx(0.5)


def test_polygon_only_covers_cells_it_overlaps():
    elevation = np.array([[-50.0, -50.0], [-50.0, -50.0]], dtype=np.float32)
    source_code = np.full((2, 2), bed.SOURCE_INVERSION, dtype=np.float32)

    stat = volume._sample_polygon(
        elevation, source_code, _transform(), shapely.box(0, 0, 1, 1)
    )

    assert stat["n_cells"] == 1
    assert stat["mean_depth_m"] == pytest.approx(50.0)


def test_thin_sliver_between_cell_centres_still_gets_a_volume():
    """The case cell-centre sampling dropped: a strip on a cell boundary.

    The transform's row 0 spans y in [1, 2] and row 1 spans [0, 1]; the strip
    y in [0.999, 1.001] contains neither row's centre, but overlaps both.
    """
    elevation = np.array([[-100.0, -100.0], [-200.0, -200.0]], dtype=np.float32)
    source_code = np.full((2, 2), bed.SOURCE_INVERSION, dtype=np.float32)

    stat = volume._sample_polygon(
        elevation, source_code, _transform(), shapely.box(0.0, 0.999, 2.0, 1.001)
    )

    # Two columns x two rows, 0.001 m^2 overlap in each of the four cells.
    assert stat["n_cells"] == 4
    assert stat["n_valid"] == 4
    # 2 m x 0.001 m overlap per row: 0.002 * 100 + 0.002 * 200.
    assert stat["volume_change_m3"] == pytest.approx(0.6)
    assert stat["mean_depth_m"] == pytest.approx(150.0)


def test_depth_is_weighted_by_partial_cell_overlap():
    """Half a cell counts half; the covered area is not assumed to be the polygon area."""
    elevation = np.array([[-100.0, -100.0], [-200.0, -200.0]], dtype=np.float32)
    source_code = np.full((2, 2), bed.SOURCE_INVERSION, dtype=np.float32)
    # Full row 0 (2 m^2) and half of row 1 (1 m^2).
    polygon = shapely.box(0.0, 0.5, 2.0, 2.0)

    stat = volume._sample_polygon(elevation, source_code, _transform(), polygon)

    assert stat["n_cells"] == 4
    assert stat["volume_change_m3"] == pytest.approx(2.0 * 100.0 + 1.0 * 200.0)
    assert stat["mean_depth_m"] == pytest.approx(400.0 / 3.0)


def test_volume_product_has_the_ensemble_columns():
    assert volume.VOLUME_PRODUCT_PATH.name == "submarine_volume_change.arrow"
    columns = volume._output_columns()
    for expected in (
        "svift_v11_mean_depth_m",
        "svift_v10_volume_change_m3",
        "vanpelt_interpolated_share",
        "ensemble_mean_volume_change_m3",
        "ensemble_model_count",
        "geometry",
    ):
        assert expected in columns
