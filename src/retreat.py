"""Dissolved NPI annual coastlines, and the year pairs the change polygons use.

`load_coastline` is the one shared input: `extraction.py` clips it to glacier
outlines to build per-year ice extents, and `bed.marine_domain` uses it to
decide which cells were ever open water.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

import geopandas as gpd
import shapely

from src import config, sources

ChangeType: TypeAlias = Literal["retreat", "advance"]

#: (year_from, year_to) pairs the change polygons compare. "Retreat" is land in
#: year_from that's gone by year_to; "advance" is the reverse.
COASTLINE_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (2012, 2018),
    (2018, 2023),
    (2012, 2023),
)


def load_coastline(year: int, force_redo: config.ForceRedo = False) -> shapely.Geometry:
    """Dissolved NPI coastline (all land, including glacier ice) for `year`.

    Parameters
    ----------
    year
        One of `config.COASTLINE_YEARS`.
    force_redo
        ``False`` reuses an existing cache; ``"self"``/``"all"`` both
        recompute here (this cache has no derived-cache dependencies -- the
        underlying download is refreshed separately via
        `sources.get_coastline`'s `refresh_download`).

    Returns
    -------
    shapely.Geometry
        Single (Multi)Polygon, EPSG:32633, the union of every land polygon
        in the year's coastline shapefile.
    """
    cache_path = config.CACHE_DIR / f"coastline_dissolved_{year}.arrow"

    if cache_path.is_file() and not force_redo:
        return gpd.read_feather(cache_path).geometry.iloc[0]

    zip_path = sources.get_coastline(year)
    coast = gpd.read_file(f"zip://{zip_path}").to_crs(config.CRS_EPSG)
    dissolved = shapely.union_all(coast.geometry.values)

    out = gpd.GeoDataFrame({"year": [year]}, geometry=[dissolved], crs=config.CRS_EPSG)

    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp.arrow")
    try:
        out.to_feather(tmp_path)
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return gpd.read_feather(cache_path).geometry.iloc[0]
