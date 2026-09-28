"""RGI7 outline loading and correction for Svalbard tidewater glaciers.

Ports the outline-correction logic from erikmannerfelt/ad_svalbard_post
(``src/outlines.py::make_outlines``) with two deliberate simplifications: this
project needs one static corrected-outline set rather than upstream's
per-interval outlines, so both the per-interval coastline clipping and the
``first_active_year``/``last_active_year`` correction filter are dropped. See
:func:`apply_outline_corrections` for what the dropped filter costs.
"""

from __future__ import annotations

from typing import Final

import geopandas as gpd

from src import config, sources

#: Jan Mayen is o2region "07-02" and is out of scope (see README).
SVALBARD_O2REGION: Final = "07-01"

#: Per-glacier attribute overrides applied before the tidewater filter and
#: the geometry corrections. Ported from ad_svalbard_post's field_overrides
#: dict, restricted to what's relevant without the surge-database fields
#: (S_Onset/S_Term/S_Direct/... don't exist in the plain RGI7.0 release we
#: use). Also dropped: upstream's non-surge `glac_name` cleanups (e.g.
#: "Lilliehøøkbreen", the unnamed "Austfonna, -" basins), so the output
#: `glac_name` is the plain RGI7.0 value for those glaciers.
FIELD_OVERRIDES: Final[dict[str, dict[str, object]]] = {
    # Advanced into water in early 2022; RGI7's term_type=9 (not assigned)
    # and zmin_m=5.5 m already puts it under the tidewater filter's zmin_m
    # threshold, but force term_type=1 explicitly so the flag doesn't
    # depend on that coincidence. The name "Scheelebreen" is shared by two
    # other, much smaller RGI IDs (RGI2000-v7.0-G-07-00742/-00743) that are
    # not this glacier.
    "RGI2000-v7.0-G-07-00741": {"term_type": 1},
}

#: term_type == 1 is RGI7's marine-terminating code; zmin_m <= 25 catches
#: glaciers RGI leaves at term_type == 9 (not assigned) but whose terminus
#: elevation implies a marine/tidewater front.
TIDEWATER_ZMIN_THRESHOLD_M: Final = 25.0


def load_raw_rgi7() -> gpd.GeoDataFrame:
    """Read the RGI7.0 region-07 release, reprojected and Jan-Mayen-filtered.

    Returns
    -------
    GeoDataFrame
        One row per RGI7 glacier outline in Svalbard (Jan Mayen excluded),
        in ``config.CRS_EPSG`` (EPSG:32633), with invalid rings fixed via a
        zero-width buffer (matches ad_svalbard_post).
    """
    zip_path = sources.get_rgi7()
    gdf = gpd.read_file(f"zip://{zip_path}")
    gdf = gdf.loc[gdf["o2region"] == SVALBARD_O2REGION].to_crs(config.CRS_EPSG)
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf.reset_index(drop=True)


def apply_field_overrides(rgi7: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply :data:`FIELD_OVERRIDES` to a copy of `rgi7`, keyed by `rgi_id`."""
    rgi7 = rgi7.copy()
    for rgi_id, overrides in FIELD_OVERRIDES.items():
        idx = rgi7.index[rgi7["rgi_id"] == rgi_id]
        if idx.empty:
            raise KeyError(f"field override target not found in RGI7: {rgi_id}")
        for column, value in overrides.items():
            rgi7.loc[idx, column] = value
    return rgi7


def load_outline_corrections() -> gpd.GeoDataFrame:
    """Read the vendored ``shapes/outline_corrections.geojson``.

    Adds a boolean ``is_partition_override`` column (True when `fix_type`
    contains "partition_override", possibly combined with an extension type
    via "+"), used to apply plain extensions before partition overrides.
    """
    corrections = gpd.read_file(config.OUTLINE_CORRECTIONS_PATH).to_crs(config.CRS_EPSG)
    corrections["is_partition_override"] = corrections["fix_type"].str.contains(
        "partition_override"
    )
    return corrections


def apply_outline_corrections(
    rgi7: gpd.GeoDataFrame, corrections: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Apply manual geometry corrections to RGI7 outlines.

    For each correction row (plain extensions first, then partition
    overrides, each group ordered by ascending `priority`): union the
    correction geometry into its target `rgi_id`'s outline, and subtract it
    from any other outline it geometrically overlaps. Ported from
    ad_svalbard_post's `make_outlines`, minus the per-interval
    `first_active_year`/`last_active_year` filtering. Upstream applies a
    correction only to the intervals its window overlaps; a single static set
    cannot. Two of the 94 corrections start in 2019 rather than 2012, but both
    were checked and are inert for the outputs: Scheelebreen's added area is
    land in every study year, so it yields no change polygon, and
    Vallåkrabreen's target is not tidewater and overlaps no other outline. The
    other 92 cover the full period.

    Parameters
    ----------
    rgi7
        Output of :func:`load_raw_rgi7` (optionally after
        :func:`apply_field_overrides`).
    corrections
        Output of :func:`load_outline_corrections`.

    Returns
    -------
    GeoDataFrame
        Copy of `rgi7` with corrected geometries and an added boolean
        `outline_modified` column.
    """
    rgi7 = rgi7.copy()
    rgi7["outline_modified"] = False
    rgi_id_to_idx = rgi7.reset_index().set_index("rgi_id")["index"].to_dict()

    ordered_corrections = corrections.sort_values(["is_partition_override", "priority"])
    for _, item in ordered_corrections.iterrows():
        target_idx = rgi_id_to_idx.get(item["rgi_id"])
        if target_idx is None:
            raise KeyError(
                f"outline correction target not found in RGI7: {item['rgi_id']} (fix_id={item['fix_id']})"
            )

        overlaps = rgi7["geometry"].overlaps(item["geometry"])
        rgi7.loc[overlaps, "geometry"] = rgi7.loc[overlaps, "geometry"].difference(
            item["geometry"]
        )
        rgi7.loc[target_idx, "geometry"] = rgi7.loc[target_idx, "geometry"].union(
            item["geometry"]
        )
        rgi7.loc[overlaps, "outline_modified"] = True
        rgi7.loc[target_idx, "outline_modified"] = True

    return rgi7


def add_tidewater_flag(rgi7: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a boolean `is_tidewater` column: `term_type == 1` or low `zmin_m`."""
    rgi7 = rgi7.copy()
    rgi7["is_tidewater"] = rgi7["term_type"].eq(1) | rgi7["zmin_m"].le(
        TIDEWATER_ZMIN_THRESHOLD_M
    )
    return rgi7


def make_corrected_outlines(force_redo: config.ForceRedo = False) -> gpd.GeoDataFrame:
    """Corrected RGI7 Svalbard outlines: overrides + geometry corrections + tidewater flag.

    Parameters
    ----------
    force_redo
        ``False`` reuses an existing cache. ``"self"``/``"all"`` both
        recompute here, since this is the first cache in the chain.

    Returns
    -------
    GeoDataFrame
        One row per Svalbard RGI7 glacier, EPSG:32633, with corrected
        geometry, `outline_modified`, and `is_tidewater` columns.
    """
    cache_path = config.CACHE_DIR / "rgi7_outlines_corrected.arrow"

    if cache_path.is_file() and not force_redo:
        return gpd.read_feather(cache_path)

    rgi7 = load_raw_rgi7()
    rgi7 = apply_field_overrides(rgi7)
    corrections = load_outline_corrections()
    rgi7 = apply_outline_corrections(rgi7, corrections)
    rgi7 = add_tidewater_flag(rgi7)

    tmp_path = cache_path.with_suffix(".tmp.arrow")
    try:
        rgi7.to_feather(tmp_path)
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return gpd.read_feather(cache_path)
