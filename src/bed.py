"""Merge each bed/thickness source into a gap-free Svalbard bed surface.

Produces, per source model, a combined bed-elevation raster (m relative to
sea level, negative below) on a shared 100 m EPSG:32633 grid
(`_target_grid`, defined by SVIFT's own raster), built from four provenance
layers in fixed priority:

1. **bathymetry** -- direct-measurement IBCAO v5.2 soundings (TID-filtered)
   at sea. Preferred over an inversion there because it is a
   sounding of the present seafloor rather than a bed modelled under
   present-day or ~2010 ice.
2. **inversion** -- the source's own modelled subglacial bed/thickness,
   wherever it has one (under ice, and at sea wherever IBCAO is silent).
3. **dem** -- the NPI 50 m DTM, standing in for bed on ice-free land (rock,
   tundra -- no ice thickness to subtract, so surface *is* bed).
4. **interpolation** -- harmonic (Laplace) fill for whatever's left, mainly
   ice-covered gaps in the source model's own coverage and marine gaps
   between sparse IBCAO soundings.

Each cell's provenance is recorded in a companion `source_code` band (see
`SOURCE_INVERSION`/`SOURCE_BATHYMETRY`/`SOURCE_DEM`/`SOURCE_INTERPOLATION`),
so "confidently measured" and "smoothly guessed" never look the same on
disk even though the elevation surface itself is seamless.

The marine and land/ice-covered domains are interpolated as two *separate*
harmonic solves, not one pass over the whole grid: a marine cell's bed must
be at or below sea level by definition (the coastline says open water sat
there at least once), so any "measured" marine cell above 0 is untrusted
and excluded from that solve's boundary conditions. With every known value
the marine solve sees held to <= 0, the harmonic maximum principle
*guarantees* every interpolated marine cell is too. The land/glacier solve
gets no such restriction -- a nunatak's positive bed under ice is ordinary.

The interpolation matters more than it sounds at sea: IBCAO's
direct-measurement coverage in Svalbard's fjords is a sparse pattern of ship
tracks, so without it the merged raster would be mostly holes there. It
solves Laplace's equation with every measured cell held fixed as a Dirichlet
boundary condition; a harmonic solution obeys a maximum principle, so an
interpolated cell can never fall outside the range of the measured values
enclosing it (see `harmonic_fill`). That structurally rules out the failure
mode of an earlier scattered-point RBF approach, which could reach 10+ km
across a data gap and import mountain elevations into a fjord.

All six `BED_MODELS` go through the same treatment: `svift_v11` and
`svift_v10` (SVIFT's two published versions), Maffezzoli et al. (IceBoost
v2.0), Millan et al. (thickness only, combined with the NPI DEM), Frank et al.
(subglacial topography), and van Pelt & Frank (bed, resampled to its true
coarser model grid). Each has its own `load_<source>_crop` loader and
`make_<source>_bed` wrapper around the shared `_make_merged_bed` core.
`export_all_products` republishes every `products/bed/<model>_bed_100m.tif`,
and `export_bed_spread_product` writes the per-pixel spread over
`ENSEMBLE_MODELS` (`svift_v10` and `millan` excluded; see there).
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Final, TypeAlias

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.warp
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg
import shapely
import tqdm

from src import config, extraction, outlines, retreat, sources

#: Common interface every per-source bed loader implements, so
#: `_make_merged_bed` can be parameterized by source. `window` indexes
#: SVIFT's own raster (only `load_svift_crop` needs it, for a direct read);
#: every reprojecting loader works from `transform`/`shape` instead.
SourceCropLoader: TypeAlias = Callable[
    [rasterio.windows.Window, rasterio.Affine, tuple[int, int]], np.ndarray
]

#: IBCAO/GEBCO-shared TID codes treated as direct/measured (see 
#: / README IBCAO notes -- IBCAO's own readme confirms it reuses GEBCO's
#: TID coding scheme): singlebeam, multibeam, seismic, isolated soundings,
#: ENC soundings, lidar, optical, combination-of-direct.
TID_DIRECT_CODES: Final = frozenset({10, 11, 12, 13, 14, 15, 16, 17})

#: Buffer around the attributed change polygons
#: (`extraction.CHANGE_POLYGONS_PATH`) whose bounding box (plus `CROP_PAD_M`)
#: defines the raster extent we work on. It spans
#: essentially all of Svalbard's tidewater coast; the buffer only has to be
#: wide enough that no attributed polygon touches the crop edge.
FILL_AOI_BUFFER_M: Final = 300.0

#: Padding added around the fill AOI's bounding box, so measured data just
#: outside the AOI is still available to constrain the fill at its edge.
CROP_PAD_M: Final = 5_000.0

#: Convergence settings for the harmonic fill's conjugate-gradient solve.
#: These control only how precisely the unique solution is reached, not
#: which solution it is, so neither is a free parameter that can move a
#: published number. 1e-6 converges the ~6.5M-unknown marine solve in well
#: under the iteration cap.
CG_TOLERANCE: Final = 1e-6
CG_MAX_ITERATIONS: Final = 5000

#: `source_code` band values -- which of the four provenance layers a
#: cell's elevation came from (see module docstring). Kept as plain floats
#: (not an enum) since they're written straight into a float32 GeoTIFF
#: band alongside elevation/distance.
SOURCE_INVERSION: Final = 1.0
SOURCE_BATHYMETRY: Final = 2.0
SOURCE_DEM: Final = 3.0
SOURCE_INTERPOLATION: Final = 4.0
SOURCE_CODE_LABELS: Final[dict[float, str]] = {
    SOURCE_INVERSION: "inversion (source model bed/thickness)",
    SOURCE_BATHYMETRY: "bathymetry (IBCAO direct measurement)",
    SOURCE_DEM: "dem (NPI DTM, ice-free land)",
    SOURCE_INTERPOLATION: "interpolation (harmonic fill)",
}

#: van Pelt & Frank's `Bed_map.tif` is delivered at 100 m, but its
#: companion `Mask.tif` reveals a coarser underlying ice-flow-model grid
#: (PISM/IGM). Snapping to ~500 m with a 200 m offset before regridding
#: avoids false precision from the 100 m delivery grid; ported from
#: svalbard_radar's `get_vanpelt_lowres`, which applied the same
#: empirically-tuned resample to this dataset's thickness counterpart.
VANPELT_LOWRES_M: Final = 500.0
VANPELT_LOWRES_SHIFT_M: Final = 200.0

#: How close an inversion's bed has to sit to the DEM surface before it is
#: read as "this model thinks there is no ice here" rather than as a bed.
#: A model reporting bed == surface is reporting zero thickness, which in the
#: marine change zone contradicts the coastline that put ice or water there.
#: Found 2026-09-17 at the Nathorstbreen Glacier System, which surged ~2008:
#: the 2008 coastline records the surge ice as land, the ~2010 NPI DEM records
#: its surface at ~166 m, and Frank's mask says ice-free -- so Frank's "bed"
#: is the stale surge surface, 166 m above sea level in a fjord whose floor is
#: -350 m two cells away. 5 m is far below the 43-77 m median implied ice
#: thickness at the cells this rule screens, so it separates the degenerate
#: tail cleanly.
SURFACE_MATCH_TOLERANCE_M: Final = 5.0

#: Bed elevation is clamped to the NPI DEM surface wherever a source model
#: puts it higher. Nothing can lie above the ground it sits under, so this is
#: the land-side counterpart of the rule that a cell the coastline calls open
#: water in every observed year cannot have a bed above sea level.
#:
#: Measured 2026-09-17, cells where the source exceeds the DEM: `millan` 0
#: (it is thickness subtracted from a surface, so it is consistent by
#: construction), `maffezzoli` 0.61%, `svift_v11` 0.71%, `vanpelt` 1.49%,
#: `svift_v10` 1.60%, `frank` 12.10%. Median excess 7-14 m, reaching 414 m.
#:
#: Clamping rather than discarding, for three reasons. The excess is usually
#: small enough to be an epoch/resolution mismatch against a ~2010 DEM rather
#: than a useless value, so the cell still carries information. Discarding
#: would punch 543 000 holes in `frank` alone and replace real topography with
#: this project's own interpolation. And on ice-free land bed == surface is
#: simply correct: it states zero ice thickness.
#:
#: It also composes with `SURFACE_MATCH_TOLERANCE_M`: a clamped cell has
#: bed == surface, so inside the marine change zone it is then screened as a
#: zero-thickness assertion and re-solved from its neighbours. The clamp
#: therefore behaves as a clamp on land and as a rejection at sea, which is
#: the right answer in each case, without a second rule.
CLAMP_BED_TO_SURFACE: Final = True

#: Hand-drawn polygon marking `frank` data to discard, digitized by the user
#: on 2026-09-17 from `topg_final/RGI60-07.00001_topg.tif`, where the bad
#: values are obvious by eye. Covers the Nathorstbreen surge footprint.
FRANK_OVERRIDE_PATH: Final = config.SHAPES_DIR / "ngs_frank_override.geojson"


def _dependency_force_redo(force_redo: config.ForceRedo) -> config.ForceRedo:
    """Only `"all"` propagates into upstream derived caches; `"self"` stops here."""
    return "all" if force_redo == "all" else False


def _fill_aoi_geometry(force_redo: config.ForceRedo = False) -> shapely.Geometry:
    changes = extraction.make_rgi_attributed_changes(force_redo=force_redo)
    return shapely.union_all(changes.geometry.buffer(FILL_AOI_BUFFER_M).values)


def _crop_window(
    svift_transform: rasterio.Affine,
    svift_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> rasterio.windows.Window:
    left, bottom, right, top = bounds
    window = rasterio.windows.from_bounds(
        left - CROP_PAD_M,
        bottom - CROP_PAD_M,
        right + CROP_PAD_M,
        top + CROP_PAD_M,
        transform=svift_transform,
    )
    window = window.round_lengths().round_offsets()
    full = rasterio.windows.Window(0, 0, svift_shape[1], svift_shape[0])
    return window.intersection(full)


def _target_grid(
    force_redo: config.ForceRedo = False,
) -> tuple[rasterio.windows.Window, rasterio.Affine, tuple[int, int]]:
    """The crop window/transform/shape every bed source is regridded onto.

    SVIFT's own raster defines the grid (100 m, EPSG:32633); the window is
    the change polygons' bounding box, padded by `CROP_PAD_M` for
    interpolation context at its edges.
    """
    with rasterio.open(sources.get_svift_bed()) as svift_src:
        svift_transform, svift_shape = svift_src.transform, svift_src.shape
    window = _crop_window(
        svift_transform, svift_shape, _fill_aoi_geometry(force_redo).bounds
    )
    transform = rasterio.windows.transform(window, svift_transform)
    shape = (round(window.height), round(window.width))
    return window, transform, shape


def load_svift_crop(
    window: rasterio.windows.Window,
    transform: rasterio.Affine,
    shape: tuple[int, int],
) -> np.ndarray:
    """Read the SVIFT bed array for `window`, as float32 with NaN nodata.

    SVIFT already sits on the target grid, so this is a direct windowed
    read; `transform`/`shape` are unused (kept for interface parity with
    the other, reprojecting, `load_<source>_crop` loaders).
    """
    with rasterio.open(sources.get_svift_bed()) as src:
        return src.read(1, window=window, masked=True).filled(np.nan).astype(np.float32)


def load_svift_v10_crop(
    window: rasterio.windows.Window,
    transform: rasterio.Affine,
    shape: tuple[int, int],
) -> np.ndarray:
    """Read SVIFT 1.0's bed array for `window` (identical grid to v1.1, see `load_svift_crop`).

    Kept as a separate model rather than swapping v1.1 out: the two are
    genuinely different published versions.
    """
    with rasterio.open(sources.get_svift_v10_bed()) as src:
        return src.read(1, window=window, masked=True).filled(np.nan).astype(np.float32)


def load_ibcao_crop(
    window: rasterio.windows.Window, transform: rasterio.Affine, shape: tuple[int, int]
) -> np.ndarray:
    """IBCAO v5.2 bathymetry, TID-filtered to direct measurements, reprojected onto the SVIFT crop grid.

    The marine bathymetry layer for every `BED_MODELS` entry.
    """
    bathy_path = sources.get_ibcao_svalbard("depth")
    tid_path = sources.get_ibcao_svalbard("tid")

    with rasterio.open(bathy_path) as bathy_src, rasterio.open(tid_path) as tid_src:
        bathy = bathy_src.read(1).astype(np.float32)
        tid = tid_src.read(1)
        bathy[~np.isin(tid, list(TID_DIRECT_CODES))] = np.nan

        dst = np.full(shape, np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=bathy,
            destination=dst,
            src_transform=bathy_src.transform,
            src_crs=bathy_src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=config.CRS_EPSG,
            dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.bilinear,
        )
    return dst


def load_maffezzoli_crop(
    window: rasterio.windows.Window, transform: rasterio.Affine, shape: tuple[int, int]
) -> np.ndarray:
    """Maffezzoli et al. IceBoost v2.0 bed, mosaicked from its 32633 + 32635 UTM tiles.

    Bed = orthometric surface elevation minus modelled ice thickness, i.e.
    `(h_wgs84 - n_geoid) - thickness` (see `sources.MAFFEZZOLI_TILE_URLS`).
    The two UTM tiles are reprojected onto the target grid independently
    and combined with the native 32633 tile taking priority (no
    reprojection resampling) over the 32635 tile, which only contributes
    the strip east of the 32633 tile's own extent (Nordaustlandet/Kvitøya).
    """
    merged = np.full(shape, np.nan, dtype=np.float32)
    for epsg in (32635, 32633):  # 32633 last, so it overwrites where both cover a cell
        with rasterio.open(sources.get_maffezzoli_tile(epsg)) as src:
            bands: dict[str, np.ndarray] = {}
            for band_index, name in ((1, "thickness"), (4, "h_wgs84"), (5, "n_geoid")):
                dst = np.full(shape, np.nan, dtype=np.float32)
                rasterio.warp.reproject(
                    source=rasterio.band(src, band_index),
                    destination=dst,
                    dst_transform=transform,
                    dst_crs=config.CRS_EPSG,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=rasterio.warp.Resampling.bilinear,
                )
                bands[name] = dst
            tile_bed = (bands["h_wgs84"] - bands["n_geoid"]) - bands["thickness"]
        merged = np.where(~np.isnan(tile_bed), tile_bed, merged)
    return merged


def load_npi_dem_crop(
    window: rasterio.windows.Window, transform: rasterio.Affine, shape: tuple[int, int]
) -> np.ndarray:
    """NPI 50 m DTM (EPSG:25833), reprojected onto the target grid.

    Used both as Millan's missing surface layer (`load_millan_crop`) and,
    since 2026-09-02, as every model's ice-free-land stand-in bed (see
    module docstring) -- on land with no ice, the true bed elevation *is*
    the surface elevation.
    """
    with rasterio.open(sources.get_npi_dem()) as src:
        dst = np.full(shape, np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=transform,
            dst_crs=config.CRS_EPSG,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.bilinear,
        )
    return dst


def load_millan_crop(
    window: rasterio.windows.Window, transform: rasterio.Affine, shape: tuple[int, int]
) -> np.ndarray:
    """Millan et al. (2022) bed = NPI 50 m DTM minus ice thickness.

    Millan has no bed layer, only thickness (already EPSG:32633, mosaicked
    from its RGI-7.1 + RGI-7.2 tiles, nodata 0); the NPI DEM supplies the
    surface to subtract it from. Thickness outliers outside [0, 1000] m are
    dropped before differencing, per svalbard_radar's `sample_models`
    precedent for this same product. That guard is **inert on the current
    Svalbard tiles** -- measured maxima are 999.4 m (RGI-7.1) and 405.2 m
    (RGI-7.2), so nothing is dropped (measured 2026-09-02). Kept
    for provenance with the upstream precedent and as protection if the
    product is revised, not because it is doing work today.
    """
    thickness = np.full(shape, np.nan, dtype=np.float32)
    for tile in ("7.1", "7.2"):
        with rasterio.open(sources.get_millan_thickness(tile)) as src:
            tile_thickness = np.full(shape, np.nan, dtype=np.float32)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=tile_thickness,
                dst_transform=transform,
                dst_crs=config.CRS_EPSG,
                src_nodata=0.0,
                dst_nodata=np.nan,
                resampling=rasterio.warp.Resampling.bilinear,
            )
        thickness = np.where(~np.isnan(tile_thickness), tile_thickness, thickness)
    thickness[(thickness < 0) | (thickness > 1000)] = np.nan

    surface = load_npi_dem_crop(window, transform, shape)
    return surface - thickness


def _bounds_intersect(
    a: rasterio.coords.BoundingBox, b: tuple[float, float, float, float]
) -> bool:
    return not (a.left > b[2] or a.right < b[0] or a.bottom > b[3] or a.top < b[1])


def load_frank_crop(
    window: rasterio.windows.Window, transform: rasterio.Affine, shape: tuple[int, int]
) -> np.ndarray:
    """Frank et al. subglacial topography, mosaicked from its per-glacier-complex tiles.

    Tiles carry an implicit nodata of 0 (the rectangular footprint padding
    around each glacier's own model domain, and no tile declares a
    nodata value, hence the explicit
    `src_nodata=0.0` below. Per-glacier RGI60 IDs in the filenames are
    irrelevant here since attribution is by geometry, not by matching IDs.

    **Tiles are not all EPSG:32633** (corrected 2026-09-02): of 617 tiles,
    481 are 32633, 64 are 32634, 55 are 32632, 14 are
    32629 (Jan Mayen), 2 are 32635 and 1 is 32636; two are 200 m rather than
    100 m. The per-tile `reproject` handles that correctly since it reads
    each tile's own CRS. The bounds pre-filter, however, compares each
    tile's *native-CRS* bounds against target bounds in EPSG:32633, which is
    not a valid comparison -- it happens to select the right 603 tiles here
    only because UTM zones 32-36 have numerically overlapping ranges at
    79 deg N. The assertion below fails loudly if that coincidence ever stops
    holding, rather than silently dropping a tile that covers Svalbard.
    """
    target_bounds = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    mosaic = np.full(shape, np.nan, dtype=np.float32)
    covered = np.zeros(shape, dtype=bool)

    with zipfile.ZipFile(sources.get_frank_topg()) as zip_file:
        for name in zip_file.namelist():
            if not name.endswith(".tif"):
                continue
            with rasterio.MemoryFile(zip_file.read(name)) as mem, mem.open() as src:
                naive_hit = _bounds_intersect(src.bounds, target_bounds)
                if str(src.crs) != f"EPSG:{config.CRS_EPSG}":
                    reprojected = rasterio.warp.transform_bounds(
                        src.crs, config.CRS_EPSG, *src.bounds
                    )
                    true_hit = _bounds_intersect(
                        rasterio.coords.BoundingBox(*reprojected), target_bounds
                    )
                    if naive_hit != true_hit:
                        raise ValueError(
                            f"Frank tile {name} ({src.crs}) is selected differently by "
                            "its native-CRS bounds than by its reprojected bounds "
                            f"(naive={naive_hit}, true={true_hit}); the bounds "
                            "pre-filter can no longer be trusted -- reproject before "
                            "comparing (see this function's docstring)"
                        )
                if not naive_hit:
                    continue
                tile = np.full(shape, np.nan, dtype=np.float32)
                rasterio.warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=tile,
                    dst_transform=transform,
                    dst_crs=config.CRS_EPSG,
                    src_nodata=0.0,
                    dst_nodata=np.nan,
                    resampling=rasterio.warp.Resampling.bilinear,
                )
            new = ~np.isnan(tile) & ~covered
            mosaic[new] = tile[new]
            covered |= ~np.isnan(tile)

    return mosaic


def load_vanpelt_crop(
    window: rasterio.windows.Window, transform: rasterio.Affine, shape: tuple[int, int]
) -> np.ndarray:
    """van Pelt & Frank (2025) bed, snapped to its true ~500 m model grid before regridding.

    `Bed_map.tif` (EPSG:25833) is masked to `Mask.tif != 0` (modelled
    cells) first -- verified 2026-09-01: 79% of the raster is unmodelled-
    area fill (bed exactly 0, mask 0), while bed == 0 is essentially never
    real where the model actually ran. The masked field is then nearest-resampled to
    `VANPELT_LOWRES_M` with `VANPELT_LOWRES_SHIFT_M` (see module-level
    comment) -- nearest is deliberate for *this* step, snapping onto the
    true, coarser model grid rather than interpolating between physically
    distinct cells. The second resample, onto the 100 m target grid, uses
    bilinear instead (2026-09-02, at the user's request): that step is a
    plain regrid of an already-representative coarse field onto a finer
    grid, where nearest just produces visibly blocky 500 m tiles for no
    benefit.
    """
    with (
        rasterio.open(sources.get_vanpelt_frank_bed()) as bed_src,
        rasterio.open(sources.get_vanpelt_frank_mask()) as mask_src,
    ):
        native = bed_src.read(1)
        modelled = mask_src.read(1) != 0
        native = np.where(modelled, native, np.nan).astype(np.float32)

        left = bed_src.bounds.left - VANPELT_LOWRES_SHIFT_M
        bottom = bed_src.bounds.bottom - VANPELT_LOWRES_SHIFT_M
        right = bed_src.bounds.right - VANPELT_LOWRES_SHIFT_M
        top = bed_src.bounds.top - VANPELT_LOWRES_SHIFT_M
        lowres_shape = (
            max(round((top - bottom) / VANPELT_LOWRES_M), 1),
            max(round((right - left) / VANPELT_LOWRES_M), 1),
        )
        lowres_transform = rasterio.transform.from_origin(
            left, top, VANPELT_LOWRES_M, VANPELT_LOWRES_M
        )
        lowres = np.full(lowres_shape, np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=native,
            destination=lowres,
            src_transform=bed_src.transform,
            src_crs=bed_src.crs,
            src_nodata=np.nan,
            dst_transform=lowres_transform,
            dst_crs=bed_src.crs,
            dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.nearest,
        )

        dst = np.full(shape, np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=lowres,
            destination=dst,
            src_transform=lowres_transform,
            src_crs=bed_src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=config.CRS_EPSG,
            dst_nodata=np.nan,
            resampling=rasterio.warp.Resampling.bilinear,
        )
    return dst


def marine_domain(
    transform: rasterio.Affine,
    shape: tuple[int, int],
    force_redo: config.ForceRedo = False,
) -> np.ndarray:
    """Cells that are sea in at least one of the `config.MARINE_DOMAIN_YEARS`.

    A cell counts as marine unless it is land in *every* one of those
    years, so both the retreat zone (ice then, water now) and the advance
    zone (water then, ice now) fall inside the domain we interpolate
    bathymetry over -- and, since 2026-09-02, so does a cell drowned by a
    pre-2012 glacier advance/surge that never retreated back into view
    during 2012-2023 (`MARINE_DOMAIN_YEARS` reaches back to 2008 for
    exactly this, found via
    the Nathorstbreen Glacier System). Restricting the fill this way is
    what keeps inland terrain from bleeding into the fjords: the solve
    region simply does not extend onto land.
    """
    land_always = shapely.intersection_all(
        [
            retreat.load_coastline(year, force_redo=force_redo)
            for year in config.MARINE_DOMAIN_YEARS
        ]
    )
    land_mask = rasterio.features.geometry_mask(
        [land_always], out_shape=shape, transform=transform, invert=True
    )
    return ~land_mask


def frank_override_zone(
    transform: rasterio.Affine, shape: tuple[int, int]
) -> np.ndarray:
    """Cells where `frank`'s own bed is discarded, from `FRANK_OVERRIDE_PATH`.

    The Nathorstbreen system surged ~2008 and `frank`'s ice mask still treats
    the resulting fjord as ice-free land, so its "bed" there is the stale
    surge surface: measured 2026-09-17, up to +200 m over 1508 marine cells
    whose floor is about -350 m. The pattern is obvious in the source raster
    and was outlined by hand.

    A hand-drawn polygon rather than a rule, on purpose. Every geometric
    proxy tried first either missed most of it or caught real data elsewhere:
    the excess is not a rim around the 2008 margin (a 100 m band covers 13.6%
    of the affected cells, 1000 m covers 97%), and the cells are not close
    enough to the surface for `SURFACE_MATCH_TOLERANCE_M` to see them. This is
    a known defect in one input over one glacier system, so it is recorded as
    such rather than generalized into a threshold that would quietly discard
    thin ice elsewhere.

    Discards the source outright rather than demoting it, so the cells fall
    through to IBCAO at sea and to the DEM or the fill on land, exactly as
    they would if `frank` had no data there. That also covers the cells just
    inland of the marine domain, which no marine-side rule can reach.
    """
    zone = gpd.read_file(FRANK_OVERRIDE_PATH).to_crs(config.CRS_EPSG)
    return rasterio.features.geometry_mask(
        zone.geometry.values, out_shape=shape, transform=transform, invert=True
    )


def always_sea_domain(
    transform: rasterio.Affine,
    shape: tuple[int, int],
    force_redo: config.ForceRedo = False,
) -> np.ndarray:
    """Cells that are sea in *every* `config.MARINE_DOMAIN_YEARS`.

    The strict complement of `marine_domain`'s "sea in at least one year":
    here the coastline product asserts open water at every epoch it
    observed, which is a positive statement that nothing solid was there.
    `marine_domain` minus this is the change zone, ground that was land or
    ice at some point and is therefore allowed an above-sea-level bed.
    """
    land_ever = shapely.union_all(
        [
            retreat.load_coastline(year, force_redo=force_redo)
            for year in config.MARINE_DOMAIN_YEARS
        ]
    )
    land_mask = rasterio.features.geometry_mask(
        [land_ever], out_shape=shape, transform=transform, invert=True
    )
    return ~land_mask


def glacier_domain(
    transform: rasterio.Affine,
    shape: tuple[int, int],
    force_redo: config.ForceRedo = False,
) -> np.ndarray:
    """Every RGI7 (Svalbard) glacier outline, rasterized -- all currently-ice-covered cells.

    Splits the non-marine part of the grid into ice-covered (bed unknown
    without a source model or interpolation) and ice-free land (bed ==
    surface, safe to fill from the NPI DEM -- see module docstring and
    `_make_merged_bed`). Every ice-free-land cell gets its real DEM elevation,
    which anchors the adjacent marine harmonic fill more accurately than the
    synthetic flat waterline this replaced.
    """
    rgi7 = outlines.make_corrected_outlines(force_redo=force_redo)
    return rasterio.features.geometry_mask(
        rgi7.geometry.values, out_shape=shape, transform=transform, invert=True
    )


def harmonic_fill(
    values: np.ndarray, known: np.ndarray, domain: np.ndarray
) -> np.ndarray:
    """Solve Laplace's equation on `domain & ~known`, holding `known` cells fixed.

    Assembles the 5-point Laplacian over the unknown cells as a sparse
    system and solves it with conjugate gradients. The matrix is symmetric
    positive semi-definite (a graph Laplacian with Dirichlet rows folded
    into the right-hand side), which is what makes CG applicable.

    **This is a deliberate simplification of IBCAO's own gridding, not an
    independent invention.** IBCAO v5.2 fills its gaps with GMT's `surface`
    (Smith & Wessel, 1990), which solves

        (1 - T) * grad^4 z  -  T * grad^2 z = 0

    for a tension factor T in [0, 1], at T = 0.34 (Jakobsson et al. 2024,
    methods step 6). We solve the T = 1 member of that same family, which is
    exactly Laplace's equation. Choosing the endmember simplifies the problem
    in three ways at once: 1) the biharmonic term disappears, so the operator
    is the 5-point Laplacian rather than a 13-point stencil needing boundary
    conditions on derivatives, 2) the solution gains a maximum principle, so
    an interpolated cell can never fall outside the range of the measured
    values enclosing it, and 3) the resulting matrix is symmetric positive
    semi-definite, which is what makes a plain CG solve sufficient.

    Property 2 is the one that matters here. Tension exists in that family
    precisely to suppress the overshoot of the minimum-curvature (T = 0)
    limit, and taking tension to its maximum removes that failure mode
    entirely rather than damping it. It is the failure that dominated an
    earlier scattered-point RBF approach, which reached 10+ km across a data
    gap and imported mountain elevations into a fjord.

    We also apply it once, at the native 100 m grid, where IBCAO solves at
    2 km, smooths with a 6 km cosine kernel, resamples bicubically to 100 m
    and then merges high-resolution data back through a remove-and-restore
    difference field. Their extra stages exist to blend survey datasets of
    very different resolution, which is not a problem we have: we consume
    their published grid, already reduced to one value per cell.

    Note what is *not* claimed. This does not reproduce IBCAO's values, and
    is not meant to. Inside the change polygons, 80.7% of cells are land in
    IBCAO's own type-identifier grid, because they were still ice-covered in
    the compilation it was built from, and only 6.0% carry a direct
    measurement. The fill therefore covers ground IBCAO does not treat as
    ocean at all; the relationship is one of method lineage, not of numerical
    agreement.

    `CG_TOLERANCE` affects only how precisely the unique solution is
    reached, not which solution it is; there is no free parameter here that
    changes the answer.

    Parameters
    ----------
    values
        Bed elevations; entries where `known` is False are ignored as input.
    known
        Cells held fixed as Dirichlet boundary conditions.
    domain
        Cells the solution is defined on. Cells outside it never contribute,
        which is how land is excluded from the marine interpolation.

    Returns
    -------
    ndarray
        `values` with every `domain & ~known` cell replaced by the harmonic
        interpolant. Cells outside `domain` are returned unchanged. Domain
        components containing no `known` cell are left at zero and should be
        removed with `_drop_unconstrained_components`.

    Raises
    ------
    RuntimeError
        If the solver fails to converge within `CG_MAX_ITERATIONS`.
    """
    solve = domain & ~known
    count = int(solve.sum())
    out = values.copy().astype(np.float32)
    if count == 0:
        return out

    index = np.full(values.shape, -1, dtype=np.int64)
    index[solve] = np.arange(count)
    rows, cols = np.nonzero(solve)
    centre = index[rows, cols]

    degree = np.zeros(count)
    rhs = np.zeros(count)
    off_rows: list[np.ndarray] = []
    off_cols: list[np.ndarray] = []
    for row_step, col_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        near_row, near_col = rows + row_step, cols + col_step
        on_grid = (
            (near_row >= 0)
            & (near_row < values.shape[0])
            & (near_col >= 0)
            & (near_col < values.shape[1])
        )
        near_row, near_col = near_row[on_grid], near_col[on_grid]
        owner = centre[on_grid]

        inside = domain[near_row, near_col]
        near_row, near_col, owner = near_row[inside], near_col[inside], owner[inside]

        np.add.at(degree, owner, 1.0)
        fixed = known[near_row, near_col]
        np.add.at(rhs, owner[fixed], values[near_row[fixed], near_col[fixed]])
        off_rows.append(owner[~fixed])
        off_cols.append(index[near_row[~fixed], near_col[~fixed]])

    # A cell with no in-domain neighbour would give an all-zero row; give it a
    # unit diagonal so the system stays non-singular. Its value is arbitrary
    # and `_drop_unconstrained_components` discards it.
    degree = np.maximum(degree, 1.0)

    off_row = np.concatenate(off_rows)
    off_col = np.concatenate(off_cols)
    matrix = scipy.sparse.coo_matrix(
        (-np.ones(off_row.size), (off_row, off_col)), shape=(count, count)
    ).tocsr() + scipy.sparse.diags(degree)

    solution, info = scipy.sparse.linalg.cg(
        matrix, rhs, rtol=CG_TOLERANCE, maxiter=CG_MAX_ITERATIONS
    )
    if info != 0:
        raise RuntimeError(
            f"harmonic fill did not converge (cg info={info}, {count} unknowns); "
            f"raise CG_MAX_ITERATIONS or relax CG_TOLERANCE"
        )

    out[solve] = solution.astype(np.float32)
    return out


def _drop_unconstrained_components(
    values: np.ndarray, known: np.ndarray, domain: np.ndarray
) -> np.ndarray:
    """NaN out connected domain components containing no known cell.

    Such a component has no boundary condition anywhere, so whatever the
    relaxation leaves there is an artefact of the initial guess, not an
    interpolation of anything.
    """
    labels, count = scipy.ndimage.label(domain)
    constrained = np.zeros(count + 1, dtype=bool)
    constrained[np.unique(labels[known])] = True
    return np.where(domain & constrained[labels], values, np.nan)


def _write_bed_cache(
    cache_path: Path,
    elevation: np.ndarray,
    distance: np.ndarray,
    source_code: np.ndarray,
    transform: rasterio.Affine,
) -> None:
    """Atomically write a 3-band (elevation, distance-to-measurement, source_code) bed GeoTIFF."""
    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp.tif")
    try:
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=elevation.shape[0],
            width=elevation.shape[1],
            count=3,
            dtype="float32",
            crs=config.CRS_EPSG,
            transform=transform,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
        ) as dst:
            dst.write(elevation.astype(np.float32), 1)
            dst.write(distance.astype(np.float32), 2)
            dst.write(source_code.astype(np.float32), 3)
            dst.set_band_description(1, "bed elevation (m, negative below sea level)")
            dst.set_band_description(2, "distance to nearest measured cell (m)")
            dst.set_band_description(
                3,
                "source_code: "
                + ", ".join(
                    f"{v:.0f}={label}" for v, label in SOURCE_CODE_LABELS.items()
                ),
            )
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def bed_cache_path(model: str) -> Path:
    """Cache path for one bed model's 3-band GeoTIFF, `cache/bed_<model>.tif`."""
    return config.CACHE_DIR / f"bed_{model}.tif"


def read_cached_bed(
    model: str, force_redo: config.ForceRedo = False
) -> tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    """Read one bed model's cached raster as `(elevation, source_code, transform)`.

    Elevation and `source_code` are bands 1 and 3 (see `SOURCE_CODE_LABELS`);
    band 2 (distance to the nearest measured cell) is not returned. The cache
    is built first if it is missing or `force_redo` asks for a recompute, so
    callers get a valid raster without knowing about `BED_MODELS` builders.

    Parameters
    ----------
    model
        A key of `BED_MODELS`.
    force_redo
        ``False`` reuses the cache; ``"self"``/``"all"`` recompute through the
        model's builder (``"all"`` also propagates to its dependencies).

    Returns
    -------
    (ndarray, ndarray, Affine)
        Elevation in m (negative below sea level), provenance code, and the
        shared EPSG:32633 transform. Both arrays are float32 with NaN nodata.
    """
    if model not in BED_MODELS:
        raise KeyError(f"model must be one of {sorted(BED_MODELS)}, got {model!r}")
    BED_MODELS[model](force_redo)

    with rasterio.open(bed_cache_path(model)) as src:
        elevation = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        source_code = src.read(3, masked=True).filled(np.nan).astype(np.float32)
        transform = src.transform
    return elevation, source_code, transform


def _make_merged_bed(
    cache_path: Path,
    load_source_crop: SourceCropLoader,
    force_redo: config.ForceRedo,
    apply_frank_override: bool = False,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Shared core for every bed model: inversion under ice, bathymetry at sea, DEM on land, fill the rest.

    Band 1 of the cached GeoTIFF is bed elevation (gap-free across the
    whole crop); band 2 is the distance to the nearest directly-measured
    cell (inversion, bathymetry, or DEM -- not interpolation); band 3 is
    the `SOURCE_CODE_LABELS` provenance code. At sea IBCAO wins wherever it
    has a sounding: it measures the present seafloor, whereas every
    subglacial bed model here is modelled under present-day or ~2010 ice.

    Parameters
    ----------
    cache_path
        Where the 3-band GeoTIFF is cached.
    load_source_crop
        One of the module's `load_<source>_crop` functions.
    force_redo
        ``False`` reuses the cached GeoTIFF. ``"self"`` recomputes here
        from cached upstream outputs; ``"all"`` also recomputes the
        coastline/outline/attribution dependencies.
    apply_frank_override
        Discard the source inside `frank_override_zone` (frank only).

    Returns
    -------
    (ndarray, Affine)
        The bed array (m, negative below sea level; essentially no
        remaining NaN cells within the crop -- a tiny residual is possible
        where a whole connected marine component has no cell with a
        trustworthy (<= 0) value anywhere in it to interpolate from) and its
        affine transform, EPSG:32633.
    """
    if cache_path.is_file() and not force_redo:
        with rasterio.open(cache_path) as src:
            return src.read(1, masked=True).filled(np.nan), src.transform

    upstream = _dependency_force_redo(force_redo)
    window, transform, shape = _target_grid(upstream)

    source = load_source_crop(window, transform, shape).astype(np.float32)
    if apply_frank_override:
        source[frank_override_zone(transform, shape)] = np.nan
    bathymetry = load_ibcao_crop(window, transform, shape)
    dem = load_npi_dem_crop(window, transform, shape)

    marine = marine_domain(transform, shape, upstream)
    glacier = glacier_domain(transform, shape, upstream) & ~marine
    land_only = ~marine & ~glacier

    has_source = ~np.isnan(source)
    has_bathymetry = ~np.isnan(bathymetry)
    has_dem = ~np.isnan(dem)

    elevation = np.full(shape, np.nan, dtype=np.float32)
    source_code = np.full(shape, np.nan, dtype=np.float32)

    def _assign(cell_mask: np.ndarray, values: np.ndarray, code: float) -> None:
        elevation[cell_mask] = values[cell_mask]
        source_code[cell_mask] = code

    _assign(marine & has_bathymetry, bathymetry, SOURCE_BATHYMETRY)
    _assign(marine & ~has_bathymetry & has_source, source, SOURCE_INVERSION)
    _assign(land_only & has_source, source, SOURCE_INVERSION)
    _assign(land_only & ~has_source & has_dem, dem, SOURCE_DEM)
    _assign(glacier & has_source, source, SOURCE_INVERSION)

    # No bed can sit above the surface it lies under. See
    # `CLAMP_BED_TO_SURFACE` for why this clamps rather than discards, and for
    # how it turns into a rejection inside the marine change zone.
    if CLAMP_BED_TO_SURFACE:
        above_surface = has_dem & ~np.isnan(elevation) & (elevation > dem)
        elevation[above_surface] = dem[above_surface]

    measured = ~np.isnan(elevation)

    # Where the coastline reports open water in *every* observed year, a bed
    # above sea level cannot be right, and such a cell is contamination
    # rather than data (seen 2026-09-02: a Frank tile's own model bleeding
    # past the true ice edge, and DEM-anchored land bleeding into narrow
    # channels once the fill domain became the whole grid). Those are
    # excluded from the marine solve's boundary conditions and re-derived
    # from genuinely-marine neighbours.
    #
    # Inside the coastline's reach the assertion runs the other way: the
    # cell was land or ice at some epoch, so the bed under it may perfectly
    # well sit above sea level (ice-cored moraine, a sill, a bedrock
    # threshold) just as it may sit below (grounded below sea level). An
    # inversion model is entitled to say so, and until 2026-09-16 it could
    # not: a blanket `elevation <= 0` gate discarded every such value and
    # relabelled it interpolation. That was measured to inflate submarine
    # volume by 0.06% (millan) to 1.72% (svift_v10), always upward, since
    # `max(0, -bed)` scores a positive bed as zero depth while its
    # interpolated replacement is pulled negative by marine neighbours. At
    # Hinlopenbreen it turned van Pelt & Frank's complete coverage into a
    # reported 39.1% "interpolated".
    #
    # Only inversion values earn the exemption. Bathymetry and DEM cells
    # above sea level in the marine domain are the contamination the gate
    # was built for, whichever side of the coastline they fall on.
    #
    # Nathorstbreen is the case that makes the 2008 coastline load-bearing
    # here: its surge put ice far down the fjord, so a large area is "ever
    # land" and now ungated. 
    #
    # The land/glacier solve gets no restriction at all -- a nunatak's
    # positive bed under ice is ordinary. The two solves stay separate (not
    # one `domain=everywhere` pass) so a land cell can never act as a
    # Dirichlet neighbour for a marine gap.
    always_sea = always_sea_domain(transform, shape, upstream)
    # ... but a model whose bed coincides with the surface is asserting no ice,
    # not an above-sea-level bed, and in the change zone that contradicts the
    # coastline. See `SURFACE_MATCH_TOLERANCE_M`: at the Nathorstbreen system
    # this is the difference between a 166 m "bed" and a fjord.
    tracks_surface = has_dem & (np.abs(elevation - dem) < SURFACE_MATCH_TOLERANCE_M)
    legitimate_positive = (
        marine
        & ~always_sea
        & (source_code == SOURCE_INVERSION)
        & (elevation > 0.0)
        & ~tracks_surface
    )
    marine_known = measured & marine & ((elevation <= 0.0) | legitimate_positive)
    other_known = measured & ~marine

    filled = elevation.copy()
    final_known = np.zeros(shape, dtype=bool)

    marine_filled = harmonic_fill(
        np.where(marine_known, elevation, 0.0), marine_known, marine
    )
    marine_filled = _drop_unconstrained_components(marine_filled, marine_known, marine)

    # Admitting positive known cells costs the maximum principle: a +50 m bed
    # at a fjord head propagates outward through the solve, and measured
    # 2026-09-16 it put up to +141 m into permanently-open water where IBCAO
    # has no sounding to pin it down (19 091 cells for svift_v10). The
    # coastline's assertion still holds out there, so the interpolated result
    # is clipped back into range. Only *interpolated* always-sea cells are
    # touched; measured cells are already <= 0 by `marine_known`, and the
    # change zone is left alone, which is the whole point of the exemption.
    # The discontinuity this can leave sits exactly on the coastline, which is
    # a real boundary rather than an artifact of the grid.
    out_of_range = always_sea & marine & ~marine_known & (marine_filled > 0.0)
    marine_filled[out_of_range] = 0.0
    filled[marine] = marine_filled[marine]
    final_known[marine] = marine_known[marine]

    other_domain = ~marine
    other_filled = harmonic_fill(
        np.where(other_known, elevation, 0.0), other_known, other_domain
    )
    other_filled = _drop_unconstrained_components(
        other_filled, other_known, other_domain
    )
    filled[other_domain] = other_filled[other_domain]
    final_known[other_domain] = other_known[other_domain]

    # The solves know nothing about the surface, so an interpolated cell can
    # overshoot above it even when every boundary value respects the bound
    # (measured 2026-09-17, before this clamp: up to +124.6 m over 50 399 cells
    # for vanpelt). The constraint is a property the published raster should
    # have, not merely one its inputs have, so it is reapplied to the filled
    # field. Measured cells are already at or below the surface and are
    # unaffected.
    if CLAMP_BED_TO_SURFACE:
        overshoot = has_dem & ~np.isnan(filled) & (filled > dem)
        filled[overshoot] = dem[overshoot]

    final_source_code = np.where(
        final_known,
        source_code,
        np.where(~np.isnan(filled), SOURCE_INTERPOLATION, np.nan),
    ).astype(np.float32)

    pixel_size = abs(transform.a)
    distance = scipy.ndimage.distance_transform_edt(~final_known) * pixel_size
    distance = np.where(np.isnan(filled), np.nan, distance)

    _write_bed_cache(cache_path, filled, distance, final_source_code, transform)

    with rasterio.open(cache_path) as src:
        return src.read(1, masked=True).filled(np.nan), src.transform


def make_svift_v11_bed(
    force_redo: config.ForceRedo = False,
) -> tuple[np.ndarray, rasterio.Affine]:
    """SVIFT 1.1 + TID-filtered IBCAO bed, harmonically filled across the marine domain."""
    return _make_merged_bed(bed_cache_path("svift_v11"), load_svift_crop, force_redo)


def make_svift_v10_bed(
    force_redo: config.ForceRedo = False,
) -> tuple[np.ndarray, rasterio.Affine]:
    """SVIFT 1.0 + TID-filtered IBCAO bed. Both versions are kept; see `load_svift_v10_crop`."""
    return _make_merged_bed(
        bed_cache_path("svift_v10"), load_svift_v10_crop, force_redo
    )


def make_maffezzoli_bed(
    force_redo: config.ForceRedo = False,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Maffezzoli et al. IceBoost v2.0 + TID-filtered IBCAO bed. See `_make_merged_bed`."""
    return _make_merged_bed(
        bed_cache_path("maffezzoli"), load_maffezzoli_crop, force_redo
    )


def make_millan_bed(
    force_redo: config.ForceRedo = False,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Millan et al. (thickness - NPI DEM) + TID-filtered IBCAO bed. See `_make_merged_bed`."""
    return _make_merged_bed(bed_cache_path("millan"), load_millan_crop, force_redo)


def make_frank_bed(
    force_redo: config.ForceRedo = False,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Frank et al. subglacial topography + TID-filtered IBCAO bed. See `_make_merged_bed`."""
    return _make_merged_bed(
        bed_cache_path("frank"), load_frank_crop, force_redo, apply_frank_override=True
    )


def make_vanpelt_bed(
    force_redo: config.ForceRedo = False,
) -> tuple[np.ndarray, rasterio.Affine]:
    """van Pelt & Frank low-res bed + TID-filtered IBCAO bed. See `_make_merged_bed`."""
    return _make_merged_bed(bed_cache_path("vanpelt"), load_vanpelt_crop, force_redo)


#: Call signature every `BED_MODELS` entry implements.
BedModelBuilder: TypeAlias = Callable[
    [config.ForceRedo], tuple[np.ndarray, rasterio.Affine]
]

#: Every bed model, keyed by the short name used throughout the project.
BED_MODELS: Final[dict[str, BedModelBuilder]] = {
    "svift_v11": make_svift_v11_bed,
    "svift_v10": make_svift_v10_bed,
    "maffezzoli": make_maffezzoli_bed,
    "millan": make_millan_bed,
    "frank": make_frank_bed,
    "vanpelt": make_vanpelt_bed,
}

#: The bed estimates that enter the ensemble. Both excluded models are still
#: built and sampled, so the exclusion stays reproducible:
#:
#: - `svift_v10` is superseded by SVIFT 1.1 and would double-count the SVIFT
#:   family in the ensemble spread.
#: - `millan` is nonphysical at the glacier margins, which is exactly where
#:   the change polygons sample it. Against direct IBCAO soundings within 1 km
#:   of the change polygons (cells that were ice when the models were made
#:   and are open water now), its raw bed is 3.5-3.8x too deep (RMSE 122-209
#:   m, median bias -83 to -98 m, on the 1783/788 cells shared with the other
#:   inversions), where the others sit at 0.85-2.9x. Inside the change
#:   polygons its own inversion averages 163 m deep against 59-98 m for the
#:   rest, roughly doubling every submarine volume (measured 2026-09-29).
ENSEMBLE_MODELS: Final[tuple[str, ...]] = (
    "svift_v11",
    "maffezzoli",
    "frank",
    "vanpelt",
)


def export_all_products(
    force_redo: config.ForceRedo = False,
) -> list[Path]:
    """Build and republish every `products/bed/` raster from the current caches.

    `products/bed/*.tif` are copies of the caches, so they go stale silently
    whenever a bed is rebuilt and nothing republishes them. Run this after
    any rebuild.
    """
    written = [
        export_bed_product(model, force_redo)
        for model in tqdm.tqdm(BED_MODELS, desc="exporting bed models")
    ]
    written.append(export_bed_spread_product(ENSEMBLE_MODELS, force_redo))
    return written


def export_bed_spread_product(
    models: tuple[str, ...] = ENSEMBLE_MODELS,
    force_redo: config.ForceRedo = False,
) -> Path:
    """Per-pixel spread across the bed ensemble, to `products/bed/bed_spread_100m.tif`.

    Three bands over the shared 100 m EPSG:32633 grid: 1) the population
    standard deviation of bed elevation across `models`, in m; 2) their
    mean; 3) how many models had a value at that cell. The statistic is on
    the signed elevation, not on `max(0, -bed)`, so a cell where the models
    disagree about whether the bed is above or below sea level shows that
    disagreement at full size rather than having it clipped away.

    Band 3 is what keeps band 1 honest: a standard deviation over two models
    is not comparable to one over four, and the ensemble's coverage is not
    uniform. Cells with fewer than two models are NaN in bands 1-2.

    **Read band 1 only where the models actually differ.** Most cells have a
    spread of exactly zero, because every model shares the same IBCAO
    bathymetry at sea and the same NPI DEM on ice-free land; the ensemble
    only disagrees where a subglacial inversion is doing the work. A
    grid-wide summary of this band is therefore dominated by cells that
    carry no information about model disagreement.
    """
    unknown = sorted(set(models) - set(BED_MODELS))
    if unknown:
        raise KeyError(f"unknown bed model(s): {unknown}")
    if len(models) < 2:
        raise ValueError("a spread needs at least two models")

    stack = []
    transform = None
    for model in models:
        elevation, model_transform = BED_MODELS[model](force_redo)
        if transform is None:
            transform = model_transform
        elif model_transform != transform:
            raise ValueError(f"{model} is not on the shared grid")
        stack.append(elevation)

    cube = np.stack(stack)
    count = np.sum(~np.isnan(cube), axis=0).astype(np.float32)
    enough = count >= 2

    with np.errstate(invalid="ignore"):
        mean = np.where(enough, np.nanmean(cube, axis=0), np.nan).astype(np.float32)
        spread = np.where(enough, np.nanstd(cube, axis=0), np.nan).astype(np.float32)

    out_path = config.PRODUCTS_DIR / "bed" / "bed_spread_100m.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_name(f"{out_path.stem}.tmp.tif")
    try:
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=spread.shape[0],
            width=spread.shape[1],
            count=3,
            dtype="float32",
            crs=config.CRS_EPSG,
            transform=transform,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
        ) as dst:
            dst.write(spread, 1)
            dst.write(mean, 2)
            dst.write(count, 3)
            dst.set_band_description(
                1, f"std dev of bed elevation across {len(models)} models (m)"
            )
            dst.set_band_description(2, "mean bed elevation (m)")
            dst.set_band_description(3, "number of models with a value")
            dst.update_tags(models=",".join(models))
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return out_path


def export_bed_product(
    model: str,
    force_redo: config.ForceRedo = False,
) -> Path:
    """Publish one bed model's raster to `products/bed/<model>_bed_100m.tif`.

    The cached raster is already spatially complete across the crop extent
    (effectively all of Svalbard's tidewater coast) rather than cropped to
    the change polygons, so this just copies it into `products/` with all
    three bands intact: bed elevation (m, negative below sea level),
    distance to the nearest measured cell (m), and the
    `SOURCE_CODE_LABELS` provenance code.
    """
    if model not in BED_MODELS:
        raise KeyError(f"model must be one of {sorted(BED_MODELS)}, got {model!r}")

    BED_MODELS[model](force_redo)
    cache_path = bed_cache_path(model)
    product_name = f"{model}_bed_100m.tif"

    out_path = config.PRODUCTS_DIR / "bed" / product_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_name(f"{out_path.stem}.tmp.tif")
    try:
        shutil.copyfile(cache_path, tmp_path)
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return out_path
