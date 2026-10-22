"""Retreat and advance polygons, attributed to RGI IDs as they are extracted.

For each coastline year, a glacier's own extent is the dissolved coastline
clipped to its RGI7 outline. Retreat and advance are then the two one-sided
differences of consecutive extents. Attribution is immediate rather than
reconstructed afterwards, and three properties follow by construction:
1) a change polygon belongs to exactly one glacier, because the outlines are
disjoint, 2) no area can be counted twice, and 3) non-glacier coastal change
never enters, so no noise filter is needed. Ice-free rock inside an outline is
land in both years and cancels in the difference.

This replaced a longer route (whole-Svalbard coastline differencing, a
minimum-area noise filter, buffered outlines, and a nearest-owner partition of
overlapping buffers) on 2026-09-03.

**The residual.** Clipping to unbuffered outlines drops change occurring just
outside them, 12.0 km^2 or 0.95% of attributed area: partly digitization slack
between the coastline product and RGI7, partly glaciers whose outline predates
a surge. `_residual_changes` recovers it by taking the coastline difference in
a `RESIDUAL_REACH_M` ring outside the outlines and assigning each connected
piece to its nearest glacier. Assigning whole pieces rather than splitting them
is safe here: of 1073 residual pieces, 998 have only one candidate glacier, and
whole-piece assignment misplaces 0.11 km^2, which is 0.0087% of attributed
change area. Contiguous ice caps are the safe case, since their shared divides
leave no gap for a piece to straddle; at eastern Austfonna none of the 131
residual pieces has more than one candidate.

**Geometry hygiene.** The difference/dissolve chain leaves debris that passes
`is_valid`: zero-width spikes and sub-millimetre fragments. `_clean_geometry`
snaps the output to `GEOMETRY_GRID_M` and drops parts below
`MIN_PIECE_AREA_M2`. Not merely cosmetic: one spike ran 1099 m out from its
neighbours, and since `bed._fill_aoi_geometry` takes the bounding box of these
polygons, it was extending the bed computation domain ~1 km southward.

**Known outline problems.** A few glaciers surged past their RGI7 outline and
lack a correction polygon, so most of their change arrives through the residual
rather than being attributed directly. The affected glaciers are not yet
enumerated; this should be worked through before the outlines are treated as
final.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Final

import geopandas as gpd
import pandas as pd
import shapely

from src import config, outlines, retreat

#: How far outside the RGI7 outlines to look for change the outlines do not
#: cover. At this reach the ring captures 99.99% of attributed change area
#: (measured 2026-09-03), so widening it further buys nothing.
RESIDUAL_REACH_M: Final = 150.0

#: Pieces below this are numerical dust from the geometry operations rather
#: than change. The per-glacier path needs no noise threshold of its own.
MIN_PIECE_AREA_M2: Final = 1.0

#: Coordinate grid the output geometry is snapped to, in m. Repeated
#: `shapely.difference` calls leave two kinds of debris that survive
#: `is_valid`: zero-width spikes, where a ring runs out and back along the
#: same line (one measured 1099 m long and exactly collinear with its
#: neighbours, contributing 0 m^2), and sub-millimetre disconnected
#: fragments. Snapping to a grid collapses both. 1 cm is four orders of
#: magnitude below the 100 m bed grid, so it cannot touch a real feature;
#: measured 2026-09-17 it changed total change area by +9e-6% while removing
#: 37% of the vertices and 61% of the parts.
#:
#: `buffer(0)` does not help here, since a collinear spike is already valid,
#: and neither does `simplify`: Douglas-Peucker anchors a ring at its most
#: distant vertex, which is the spike apex itself.
GEOMETRY_GRID_M: Final = 0.01

#: Columns of the change-polygon output.
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "rgi_id",
    "glac_name",
    "year_from",
    "year_to",
    "change_type",
    "area_m2",
    "geometry",
)

#: Project output (deliverable), not an internal intermediate: the attributed
#: retreat/advance polygons for every tidewater glacier and interval. Written
#: by `make_rgi_attributed_changes` and read by `bed` to frame its crop.
CHANGE_POLYGONS_PATH: Final = config.CACHE_DIR / "change_polygons_by_rgi.arrow"


def _clean_geometry(
    geometry: shapely.Geometry, min_area_m2: float = MIN_PIECE_AREA_M2
) -> shapely.Geometry:
    """Snap to `GEOMETRY_GRID_M`, keep polygonal parts, drop parts under `min_area_m2`.

    Repairs the debris left by the difference/dissolve chain: zero-width
    spikes, sub-millimetre fragments, and the stray `LineString` components
    that make some rows come back as a `GeometryCollection` rather than a
    polygon. Area-preserving to within a rounding error -- see
    `GEOMETRY_GRID_M`.
    """
    parts: list[shapely.Geometry] = []
    for part in shapely.get_parts(shapely.set_precision(geometry, GEOMETRY_GRID_M)):
        if part.geom_type == "MultiPolygon":
            parts.extend(shapely.get_parts(part))
        elif part.geom_type == "Polygon":
            parts.append(part)
    kept = [part for part in parts if part.area >= min_area_m2]
    return shapely.union_all(kept) if kept else shapely.Polygon()


def _tidewater_outlines(force_redo: config.ForceRedo = False) -> gpd.GeoDataFrame:
    rgi7 = outlines.make_corrected_outlines(force_redo=force_redo)
    return rgi7.loc[
        rgi7["is_tidewater"], ["rgi_id", "glac_name", "geometry"]
    ].reset_index(drop=True)


def _clip_to(geometry: shapely.Geometry, parts, tree) -> shapely.Geometry:
    """Intersect `geometry` with a dissolved coastline, via a spatial index.

    Intersecting against the whole Svalbard land multipolygon directly is
    prohibitively slow, so candidate parts are looked up first.
    """
    found = tree.query(geometry, predicate="intersects")
    if len(found) == 0:
        return shapely.Polygon()
    return shapely.intersection(shapely.union_all(parts[found]), geometry)


def make_glacier_extents(
    year: int, force_redo: config.ForceRedo = False
) -> gpd.GeoDataFrame:
    """Each tidewater glacier's ice extent in `year`.

    Parameters
    ----------
    year
        One of `config.COASTLINE_YEARS`.
    force_redo
        ``False`` reuses an existing cache. ``"self"`` recomputes this
        artifact; ``"all"`` also recomputes outlines and coastlines.

    Returns
    -------
    GeoDataFrame
        One row per tidewater glacier, EPSG:32633, with columns `rgi_id`,
        `glac_name`, `year`, `area_m2` and `geometry`.
    """
    cache_path = config.CACHE_DIR / f"glacier_extent_{year}.arrow"
    if cache_path.is_file() and not force_redo:
        return gpd.read_feather(cache_path)

    upstream: config.ForceRedo = "all" if force_redo == "all" else False
    tidewater = _tidewater_outlines(upstream)
    parts = shapely.get_parts(retreat.load_coastline(year, force_redo=upstream))
    tree = shapely.STRtree(parts)

    out = gpd.GeoDataFrame(
        {
            "rgi_id": tidewater["rgi_id"].values,
            "glac_name": tidewater["glac_name"].values,
            "year": year,
        },
        geometry=[_clip_to(g, parts, tree) for g in tidewater.geometry.values],
        crs=config.CRS_EPSG,
    )
    out["area_m2"] = out.geometry.area

    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp.arrow")
    try:
        out.to_feather(tmp_path)
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return gpd.read_feather(cache_path)


def _residual_changes(
    tidewater: gpd.GeoDataFrame, force_redo: config.ForceRedo
) -> gpd.GeoDataFrame:
    """Change just outside the outlines, assigned to the nearest glacier.

    See the module docstring for why assigning whole pieces is safe.
    """
    union = shapely.union_all(tidewater.geometry.values)
    ring = shapely.difference(shapely.buffer(union, RESIDUAL_REACH_M), union)

    land = {}
    for year in config.COASTLINE_YEARS:
        parts = shapely.get_parts(retreat.load_coastline(year, force_redo=force_redo))
        land[year] = _clip_to(ring, parts, shapely.STRtree(parts))

    records: list[dict[str, object]] = []
    for year_from, year_to in retreat.COASTLINE_PAIRS:
        early, late = land[year_from], land[year_to]
        for change_type, geometry in (
            ("retreat", shapely.difference(early, late)),
            ("advance", shapely.difference(late, early)),
        ):
            for part in shapely.get_parts(geometry):
                if part.area > MIN_PIECE_AREA_M2:
                    records.append(
                        {
                            "year_from": year_from,
                            "year_to": year_to,
                            "change_type": change_type,
                            "geometry": part,
                        }
                    )
    if not records:
        return gpd.GeoDataFrame(
            columns=["rgi_id", "glac_name", "year_from", "year_to", "change_type"],
            geometry=gpd.GeoSeries([], crs=config.CRS_EPSG),
        )

    pieces = gpd.GeoDataFrame(records, geometry="geometry", crs=config.CRS_EPSG)
    # A ring piece is at most RESIDUAL_REACH_M from an outline, but
    # sjoin_nearest measures boundary-to-boundary, so a generous 1.5x margin
    # absorbs rounding at the ring edge without admitting non-ring pieces.
    joined = pieces.sjoin_nearest(
        tidewater[["rgi_id", "glac_name", "geometry"]],
        how="inner",
        max_distance=RESIDUAL_REACH_M * 1.5,
    )
    # a tie places one piece equidistant from two glaciers; keeping the first
    # leaves it assigned exactly once
    joined = joined[~joined.index.duplicated(keep="first")].reset_index(drop=True)
    return joined.drop(columns=[c for c in ("index_right",) if c in joined.columns])


def make_rgi_attributed_changes(
    force_redo: config.ForceRedo = False,
) -> gpd.GeoDataFrame:
    """Retreat and advance per glacier and interval.

    Parameters
    ----------
    force_redo
        ``False`` reuses an existing cache. ``"self"`` recomputes this artifact
        from cached extents; ``"all"`` propagates to the extents, outlines and
        coastlines.

    Returns
    -------
    GeoDataFrame
        One row per (glacier, interval, change type), columns
        `OUTPUT_COLUMNS`, EPSG:32633. Geometry is dissolved over the glacier
        rather than exploded into contiguous pieces, and `area_m2` is
        recomputed from the final geometry.
    """
    cache_path = CHANGE_POLYGONS_PATH
    if cache_path.is_file() and not force_redo:
        return gpd.read_feather(cache_path)

    upstream: config.ForceRedo = "all" if force_redo == "all" else False
    tidewater = _tidewater_outlines(upstream)
    extents = {
        year: make_glacier_extents(year, force_redo=upstream).set_index("rgi_id")
        for year in config.COASTLINE_YEARS
    }

    records: list[dict[str, object]] = []
    for year_from, year_to in retreat.COASTLINE_PAIRS:
        early, late = extents[year_from], extents[year_to]
        for rgi_id, row in early.iterrows():
            geom_to = late.loc[rgi_id].geometry
            for change_type, geometry in (
                ("retreat", shapely.difference(row.geometry, geom_to)),
                ("advance", shapely.difference(geom_to, row.geometry)),
            ):
                if geometry.area > MIN_PIECE_AREA_M2:
                    records.append(
                        {
                            "rgi_id": rgi_id,
                            "glac_name": row["glac_name"],
                            "year_from": year_from,
                            "year_to": year_to,
                            "change_type": change_type,
                            "geometry": geometry,
                        }
                    )

    if records:
        direct = gpd.GeoDataFrame(records, geometry="geometry", crs=config.CRS_EPSG)
    else:
        direct = gpd.GeoDataFrame(
            columns=["rgi_id", "glac_name", "year_from", "year_to", "change_type"],
            geometry=gpd.GeoSeries([], crs=config.CRS_EPSG),
        )
    combined = pd.concat(
        [direct, _residual_changes(tidewater, upstream)], ignore_index=True
    )
    if combined.empty:
        # A GeoDataFrame with no rows cannot take its geometry column from a
        # named field, so build the empty product explicitly.
        empty = gpd.GeoDataFrame(
            columns=[c for c in OUTPUT_COLUMNS if c != "geometry"],
            geometry=gpd.GeoSeries([], crs=config.CRS_EPSG),
        )
        out = empty.loc[:, list(OUTPUT_COLUMNS)]
    else:
        out = (
            gpd.GeoDataFrame(combined, geometry="geometry", crs=config.CRS_EPSG)
            .dissolve(
                by=["rgi_id", "year_from", "year_to", "change_type"],
                aggfunc={"glac_name": "first"},
            )
            .reset_index()
        )
        out["geometry"] = [_clean_geometry(g) for g in out.geometry.values]
        out["area_m2"] = out.geometry.area
        out = out.loc[out["area_m2"] > 0, list(OUTPUT_COLUMNS)].reset_index(drop=True)

    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp.arrow")
    try:
        out.to_feather(tmp_path)
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return gpd.read_feather(cache_path)


def export_change_polygons_product(
    force_redo: config.ForceRedo = False,
) -> Path:
    """Publish the change polygons to `products/change_polygons_by_rgi.arrow`.

    The product is a copy of the cache, so it goes stale silently whenever the
    polygons are rebuilt and nothing republishes them. Run this after a
    rebuild.
    """
    make_rgi_attributed_changes(force_redo)

    out_path = config.PRODUCTS_DIR / "change_polygons_by_rgi.arrow"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_name(f"{out_path.stem}.tmp.arrow")
    try:
        shutil.copyfile(CHANGE_POLYGONS_PATH, tmp_path)
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return out_path
