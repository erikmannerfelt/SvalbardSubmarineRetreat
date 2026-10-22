"""Validation of each bed model's raw inversion against direct bathymetry at the calving fronts.

The submarine volume is the depth below sea level integrated over the change
polygons, so the question that matters is how well each model's *own*
inversion represents water depth at the glacier margin, not how well the
merged bed does (where IBCAO already wins wherever it has a sounding).

The test uses the cells that were ice when a model was made and are open,
sounded water now: every IBCAO direct-measurement cell (TID-filtered, see
`bed.load_ibcao_crop`) within `NEAR_FRONT_BUFFER_M` of any change polygon
that also has a raw value from the model. Different models cover different
cells, so the comparison is repeated on the cells shared by the chosen
subsets in `COMMON_CELL_SETS`; only those rows compare models like for like.

This is what justifies excluding `millan` from `bed.ENSEMBLE_MODELS`.
Whole-archipelago seam statistics (inversion-vs-bathymetry or
inversion-vs-land steps) do not single it out; this margin-restricted test
does.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import rasterio
import rasterio.features

from src import bed, config, extraction

#: The published validation table (small, user-facing: CSV).
VALIDATION_PRODUCT_PATH: Final = config.PRODUCTS_DIR / "bed_model_validation.csv"

#: How far from a change polygon a sounding still counts as "at the front".
NEAR_FRONT_BUFFER_M: Final = 1000.0

#: Each model's raw (pre-merge) bed loader, on the shared SVIFT crop grid.
RAW_BED_LOADERS: Final[dict[str, bed.SourceCropLoader]] = {
    "svift_v11": bed.load_svift_crop,
    "svift_v10": bed.load_svift_v10_crop,
    "maffezzoli": bed.load_maffezzoli_crop,
    "millan": bed.load_millan_crop,
    "frank": bed.load_frank_crop,
    "vanpelt": bed.load_vanpelt_crop,
}

#: Model subsets evaluated on the cells they all cover. `svift_v11` covers
#: too few sounded cells near the fronts (324) to share a useful set with
#: the others, so it is only evaluated on its own cells.
COMMON_CELL_SETS: Final[tuple[tuple[str, ...], ...]] = (
    ("maffezzoli", "millan", "vanpelt"),
    ("maffezzoli", "millan", "vanpelt", "frank"),
)

_COLUMNS: Final = (
    "cell_set",
    "model",
    "n_cells",
    "median_bias_m",
    "mean_bias_m",
    "rmse_m",
    "depth_ratio",
)


def _row(
    cell_set: str, model: str, model_bed: np.ndarray, sounding: np.ndarray
) -> dict[str, object]:
    """Error statistics of `model_bed` against `sounding` (both m, negative below sea level).

    `depth_ratio` is the summed model depth below sea level over the summed
    measured depth: the factor by which the model would scale a submarine
    volume on these cells. Bias is model minus measurement, so negative
    means the model is too deep.
    """
    diff = model_bed - sounding
    return {
        "cell_set": cell_set,
        "model": model,
        "n_cells": int(diff.size),
        "median_bias_m": float(np.median(diff)),
        "mean_bias_m": float(np.mean(diff)),
        "rmse_m": float(np.sqrt(np.mean(diff**2))),
        "depth_ratio": float(np.sum(np.maximum(0.0, -model_bed)) / np.sum(-sounding)),
    }


def make_bed_model_validation(
    force_redo: config.ForceRedo = False,
) -> pd.DataFrame:
    """Compare every raw bed inversion with IBCAO soundings near the calving fronts.

    Parameters
    ----------
    force_redo
        ``False`` reuses the change-polygon cache; ``"self"`` does too (the
        table itself is always rebuilt, as it is cheap relative to its
        inputs); ``"all"`` also recomputes the change polygons and the
        outlines/coastlines beneath them.

    Returns
    -------
    DataFrame
        One row per (cell set, model): `cell_set` is ``"own"`` for each
        model's own covered cells, or the ``+``-joined model names of a
        `COMMON_CELL_SETS` entry. The finalized `VALIDATION_PRODUCT_PATH`
        re-read from disk.
    """
    upstream: config.ForceRedo = "all" if force_redo == "all" else False
    window, transform, shape = bed._target_grid(upstream)
    changes = extraction.make_rgi_attributed_changes(upstream)

    near_front = rasterio.features.rasterize(
        ((geometry, 1) for geometry in changes.geometry.buffer(NEAR_FRONT_BUFFER_M)),
        out_shape=shape,
        transform=transform,
    ).astype(bool)
    sounding = bed.load_ibcao_crop(window, transform, shape)
    sounded = near_front & ~np.isnan(sounding) & (sounding < 0)

    raw: dict[str, np.ndarray] = {}
    for model, loader in RAW_BED_LOADERS.items():
        model_bed = loader(window, transform, shape).astype(np.float32)
        # The same NGS hand override the merged Frank bed gets.
        if model == "frank":
            model_bed[bed.frank_override_zone(transform, shape)] = np.nan
        raw[model] = model_bed

    rows = []
    for model, model_bed in raw.items():
        cells = sounded & ~np.isnan(model_bed)
        rows.append(_row("own", model, model_bed[cells], sounding[cells]))
    for models in COMMON_CELL_SETS:
        cells = sounded.copy()
        for model in models:
            cells &= ~np.isnan(raw[model])
        for model in models:
            rows.append(
                _row("+".join(models), model, raw[model][cells], sounding[cells])
            )
    result = pd.DataFrame(rows, columns=list(_COLUMNS))

    out_path = VALIDATION_PRODUCT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp.csv")
    try:
        result.to_csv(tmp_path, index=False, float_format="%.4f")
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return pd.read_csv(out_path)
