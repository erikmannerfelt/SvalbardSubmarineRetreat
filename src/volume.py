"""Submarine volume change per change polygon, sampled from every bed model.

For each attributed change polygon and each `bed.BED_MODELS` entry, this
integrates the cached bed raster on the shared 100 m grid. The depth at a cell
is ``max(0, -bed)``: a bed at or above sea level contributes zero submarine
depth, so the result is a true below-sea-level volume rather than a signed
elevation integral. The volume change is that depth integrated over the
polygon, weighted by how much of each cell the polygon covers; the reported
mean depth is the integral divided by the covered area.

Area weighting is load-bearing here. The change polygons run along calving
fronts, so many are sub-pixel or thin ribbons: a cell-centre test misses them
entirely (134 of 806 polygons, 0.033% of the change area), while counting
every touched cell would include cells the polygon barely clips and inflate
the result. `_cell_overlaps` computes the exact polygon-cell intersection area
instead, so a sub-pixel polygon still gets a volume and a large polygon is not
swelled by boundary cells.

This is the product the bed and change-polygon caches were built for. It is
self-contained: the sampled mean depths, volumes, interpolation shares and the
ensemble summary all live in one table, so no downstream reader needs the bed
rasters to get a volume. Every model is sampled, but only `bed.ENSEMBLE_MODELS`
enter the ensemble (`svift_v10` is a second version of the same SVIFT product).

Cells where a model's raster has no value -- an unconstrained interpolation
component -- carry no overlap weight and are counted in `<model>_valid_cells`,
so a thin-coverage number is never mistaken for a well-constrained one.
"""

from __future__ import annotations

import warnings
from typing import Final

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import shapely

from src import bed, config, extraction

#: The published volume table (a project deliverable, written directly, not
#: cached: it is cheap to rebuild and only meaningful next to the bed caches).
VOLUME_PRODUCT_PATH: Final = config.PRODUCTS_DIR / "submarine_volume_change.arrow"


def _output_columns() -> list[str]:
    """Column order of the volume product."""
    columns = [
        "rgi_id",
        "glac_name",
        "year_from",
        "year_to",
        "change_type",
        "area_m2",
        "n_cells",
    ]
    for model in bed.BED_MODELS:
        columns += [
            f"{model}_mean_depth_m",
            f"{model}_volume_change_m3",
            f"{model}_interpolated_share",
            f"{model}_valid_cells",
        ]
    columns += [
        "ensemble_mean_volume_change_m3",
        "ensemble_median_volume_change_m3",
        "ensemble_std_volume_change_m3",
        "ensemble_model_count",
        "geometry",
    ]
    return columns


def _cell_overlaps(
    transform: rasterio.Affine, shape: tuple[int, int], geometry: shapely.Geometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cells overlapping `geometry`, with the exact area of each overlap.

    `all_touched` rasterization finds candidate cells, then the polygon-cell
    intersection area is computed so a cell the polygon only corner-touches
    gets zero weight and is dropped. Returns `(rows, cols, overlap_area_m2)`,
    one entry per cell with a non-zero overlap.
    """
    touched = rasterio.features.geometry_mask(
        [geometry],
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )
    rows, cols = np.nonzero(touched)
    if rows.size == 0:
        return rows, cols, np.zeros(0, dtype="float64")

    upper_left_x, upper_left_y = rasterio.transform.xy(
        transform, rows, cols, offset="ul"
    )
    upper_left_x = np.asarray(upper_left_x, dtype="float64")
    upper_left_y = np.asarray(upper_left_y, dtype="float64")
    pixel_x = abs(transform.a)
    pixel_y = abs(transform.e)
    cells = shapely.box(
        upper_left_x,
        upper_left_y - pixel_y,
        upper_left_x + pixel_x,
        upper_left_y,
    )
    overlap = shapely.area(shapely.intersection(cells, geometry))
    keep = overlap > 0.0
    return rows[keep], cols[keep], overlap[keep].astype("float64")


def _sample_overlaps(
    rows: np.ndarray,
    cols: np.ndarray,
    overlap_m2: np.ndarray,
    elevation: np.ndarray,
    source_code: np.ndarray,
) -> dict[str, float]:
    """Depth/volume statistics of one bed raster over precomputed cell overlaps.

    Parameters
    ----------
    rows, cols, overlap_m2
        The polygon's cells and their overlap areas, from `_cell_overlaps`.
    elevation, source_code
        One bed model's band 1 and band 3, NaN outside the model's coverage.

    Returns
    -------
    dict
        `n_cells` (overlapping cells), `n_valid` (of those, cells with a
        value), `mean_depth_m` (area-weighted), `volume_change_m3` (the
        area-weighted depth integral) and `interpolated_share` (the
        area-weighted fraction coming from the harmonic fill). With no valid
        cell, the last three are NaN and `n_valid` is zero.
    """
    cell_elevation = elevation[rows, cols]
    valid = ~np.isnan(cell_elevation)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "n_cells": int(rows.size),
            "n_valid": 0,
            "mean_depth_m": np.nan,
            "volume_change_m3": np.nan,
            "interpolated_share": np.nan,
        }

    weight = overlap_m2[valid]
    depth = np.maximum(0.0, -cell_elevation[valid])
    volume = float(np.sum(weight * depth))
    covered = float(np.sum(weight))
    interpolated = source_code[rows, cols][valid] == bed.SOURCE_INTERPOLATION
    return {
        "n_cells": int(rows.size),
        "n_valid": n_valid,
        "mean_depth_m": volume / covered,
        "volume_change_m3": volume,
        "interpolated_share": float(np.sum(weight[interpolated]) / covered),
    }


def _sample_polygon(
    elevation: np.ndarray,
    source_code: np.ndarray,
    transform: rasterio.Affine,
    geometry: shapely.Geometry,
) -> dict[str, float]:
    """`_sample_overlaps` over a geometry, computing the cell overlaps first."""
    rows, cols, overlap_m2 = _cell_overlaps(transform, elevation.shape, geometry)
    return _sample_overlaps(rows, cols, overlap_m2, elevation, source_code)


def make_submarine_volume_product(
    force_redo: config.ForceRedo = False,
) -> gpd.GeoDataFrame:
    """Sample every bed model over every change polygon; write the volume product.

    Parameters
    ----------
    force_redo
        ``False`` reuses each bed cache and the change-polygon cache.
        ``"self"``/``"all"`` recompute each bed's builder (the product itself
        is always rebuilt); ``"all"`` also propagates to the change polygons
        and the outlines/coastlines beneath them.

    Returns
    -------
    GeoDataFrame
        One row per (glacier, interval, change type), columns
        `_output_columns()`, EPSG:32633, the finalized `VOLUME_PRODUCT_PATH`
        re-read from disk.
    """
    changes = extraction.make_rgi_attributed_changes(force_redo)
    upstream: config.ForceRedo = "all" if force_redo == "all" else False

    out = changes.loc[
        :,
        [
            "rgi_id",
            "glac_name",
            "year_from",
            "year_to",
            "change_type",
            "area_m2",
            "geometry",
        ],
    ].copy()

    if changes.empty:
        out["n_cells"] = []
        for model in bed.BED_MODELS:
            out[f"{model}_mean_depth_m"] = []
            out[f"{model}_volume_change_m3"] = []
            out[f"{model}_interpolated_share"] = []
            out[f"{model}_valid_cells"] = []
        for column in (
            "ensemble_mean_volume_change_m3",
            "ensemble_median_volume_change_m3",
            "ensemble_std_volume_change_m3",
            "ensemble_model_count",
        ):
            out[column] = []
    else:
        rasters = {
            model: bed.read_cached_bed(model, force_redo=upstream)
            for model in bed.BED_MODELS
        }
        transform = next(iter(rasters.values()))[2]
        shape = next(iter(rasters.values()))[0].shape

        samples: dict[str, dict[str, list[float]]] = {
            model: {
                "mean_depth_m": [],
                "volume_change_m3": [],
                "interpolated_share": [],
                "valid_cells": [],
            }
            for model in bed.BED_MODELS
        }
        n_cells: list[int] = []
        for geometry in changes.geometry.values:
            rows, cols, overlap_m2 = _cell_overlaps(transform, shape, geometry)
            n_cells.append(int(rows.size))
            for model, (elevation, source_code, _) in rasters.items():
                stat = _sample_overlaps(rows, cols, overlap_m2, elevation, source_code)
                samples[model]["mean_depth_m"].append(stat["mean_depth_m"])
                samples[model]["volume_change_m3"].append(stat["volume_change_m3"])
                samples[model]["interpolated_share"].append(stat["interpolated_share"])
                samples[model]["valid_cells"].append(stat["n_valid"])

        out["n_cells"] = n_cells
        for model in bed.BED_MODELS:
            out[f"{model}_mean_depth_m"] = samples[model]["mean_depth_m"]
            out[f"{model}_volume_change_m3"] = samples[model]["volume_change_m3"]
            out[f"{model}_interpolated_share"] = samples[model]["interpolated_share"]
            out[f"{model}_valid_cells"] = samples[model]["valid_cells"]

        ensemble = np.stack(
            [
                np.asarray(samples[model]["volume_change_m3"], dtype="float64")
                for model in bed.ENSEMBLE_MODELS
            ],
            axis=0,
        )
        # Rows covered by none of the ensemble models are legitimately all-NaN;
        # the reduced statistics on them are NaN too, without a warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["ensemble_mean_volume_change_m3"] = np.nanmean(ensemble, axis=0)
            out["ensemble_median_volume_change_m3"] = np.nanmedian(ensemble, axis=0)
            out["ensemble_std_volume_change_m3"] = np.nanstd(ensemble, axis=0)
        out["ensemble_model_count"] = np.sum(~np.isnan(ensemble), axis=0).astype(float)

    result = gpd.GeoDataFrame(out, geometry="geometry", crs=config.CRS_EPSG)
    result = result.loc[:, _output_columns()]

    out_path = VOLUME_PRODUCT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp.arrow")
    try:
        result.to_feather(tmp_path)
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return gpd.read_feather(out_path)
