"""Unit tests for the small pure helpers in `bed.py` and `outlines.py`."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely

from src import bed, config, outlines


# --------------------------------------------------------------- bed.py
def test_bed_cache_path_is_cache_dir_slash_model():
    assert bed.bed_cache_path("svift_v11") == config.CACHE_DIR / "bed_svift_v11.tif"


def test_every_model_has_a_distinct_cache_path():
    paths = {m: bed.bed_cache_path(m) for m in bed.BED_MODELS}
    assert len(set(paths.values())) == len(bed.BED_MODELS)


def test_spread_ensemble_is_the_four_estimates():
    assert set(bed.ENSEMBLE_MODELS).issubset(bed.BED_MODELS)
    assert "svift_v10" not in bed.ENSEMBLE_MODELS
    assert "millan" not in bed.ENSEMBLE_MODELS
    assert bed.ENSEMBLE_MODELS == (
        "svift_v11",
        "maffezzoli",
        "frank",
        "vanpelt",
    )


def test_source_code_labels_cover_every_code():
    codes = {
        bed.SOURCE_INVERSION,
        bed.SOURCE_BATHYMETRY,
        bed.SOURCE_DEM,
        bed.SOURCE_INTERPOLATION,
    }
    assert set(bed.SOURCE_CODE_LABELS) == codes
    assert len(codes) == 4


def test_tid_direct_codes_are_the_documented_range():
    """README/AGENTS document 10-17 as direct measurement."""
    assert bed.TID_DIRECT_CODES == frozenset(range(10, 18))


def test_harmonic_fill_leaves_a_constant_field_constant():
    """The simplest solve there is: a constant boundary implies a constant interior."""
    shape = (40, 40)
    domain = np.ones(shape, dtype=bool)
    known = np.zeros(shape, dtype=bool)
    known[0, :] = known[-1, :] = known[:, 0] = known[:, -1] = True
    values = np.zeros(shape, dtype=np.float32)
    values[known] = -73.5

    out = bed.harmonic_fill(values, known, domain)

    assert np.allclose(out[domain & ~known], -73.5, atol=1e-3)


def test_cg_settings_are_convergence_only():
    """Both remaining fill constants control precision, not the solution."""
    assert bed.CG_TOLERANCE < 1e-4
    assert bed.CG_MAX_ITERATIONS >= 1000


def test_bounds_intersect():
    import rasterio.coords

    a = rasterio.coords.BoundingBox(0.0, 0.0, 10.0, 10.0)
    assert bed._bounds_intersect(a, (5.0, 5.0, 15.0, 15.0))
    assert not bed._bounds_intersect(a, (20.0, 20.0, 30.0, 30.0))


# ---------------------------------------------------------- outlines.py
def test_tidewater_flag_uses_term_type_or_low_zmin():
    frame = gpd.GeoDataFrame(
        {
            "term_type": [1, 9, 9, 2],
            "zmin_m": [500.0, 5.5, 300.0, 25.0],
            "geometry": [shapely.box(0, 0, 1, 1)] * 4,
        },
        geometry="geometry",
        crs=32633,
    )
    flagged = outlines.add_tidewater_flag(frame)
    # term_type 1; low zmin; neither; zmin exactly at the threshold (inclusive)
    assert list(flagged["is_tidewater"]) == [True, True, False, True]


def test_field_overrides_raise_when_the_target_is_absent():
    frame = gpd.GeoDataFrame(
        {"rgi_id": ["other"], "term_type": [9], "geometry": [shapely.box(0, 0, 1, 1)]},
        geometry="geometry",
        crs=32633,
    )
    with pytest.raises(KeyError, match="field override target not found"):
        outlines.apply_field_overrides(frame)


def test_field_overrides_do_not_mutate_the_caller():
    ids = list(outlines.FIELD_OVERRIDES)
    frame = gpd.GeoDataFrame(
        {
            "rgi_id": ids,
            "term_type": [9] * len(ids),
            "geometry": [shapely.box(0, 0, 1, 1)] * len(ids),
        },
        geometry="geometry",
        crs=32633,
    )
    before = frame["term_type"].tolist()
    outlines.apply_field_overrides(frame)
    assert frame["term_type"].tolist() == before


def test_scheelebreen_is_forced_tidewater():
    """It turned tidewater mid-study and must not depend on RGI's zmin."""
    ids = list(outlines.FIELD_OVERRIDES)
    frame = gpd.GeoDataFrame(
        {
            "rgi_id": ids,
            "term_type": [9] * len(ids),
            "zmin_m": [999.0] * len(ids),
            "geometry": [shapely.box(0, 0, 1, 1)] * len(ids),
        },
        geometry="geometry",
        crs=32633,
    )
    flagged = outlines.add_tidewater_flag(outlines.apply_field_overrides(frame))
    assert flagged["is_tidewater"].all()


def test_only_two_outline_corrections_start_after_2012():
    """The static outline set drops upstream's per-interval active-year filter.

    Every correction but these two has been active since 2012; the two 2019
    ones (Scheelebreen, Vallakrabreen) were checked and are inert for the
    outputs (see `outlines.apply_outline_corrections`). If this set changes,
    the dropped filter has to be re-checked against the outputs.
    """
    corrections = outlines.load_outline_corrections()
    late = sorted(corrections.loc[corrections["first_active_year"] > 2012, "fix_id"])
    assert late == [739210, 796355]
