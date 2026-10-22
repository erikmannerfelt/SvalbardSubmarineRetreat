"""Repeatable downloads for external data sources.

Each `get_<source>()` function downloads (if not already present under
`input/`) and returns the local path to a canonical, citable upstream
resource, so every file under `input/` is traceable back to where it came
from and re-fetchable, rather than the product of a one-off manual
download. Pattern ported from the `get_<source>()` functions in
erikmannerfelt/svalbard_radar's `svalbardradar/comparisons.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import affine
import numpy as np
import requests

from src import config

RGI7_URL: Final = (
    "https://cluster.klima.uni-bremen.de/~fmaussion/fmaussion/misc/rgi7_data/"
    "rgi70_official/RGI2000-v7.0-G-global/RGI2000-v7.0-G-07_svalbard_jan_mayen.zip"
)

#: NPI annual coastlines dataset (data.npolar.no/dataset/21565514-...),
#: keyed by year. Each is a `Coast<year>.zip` containing a land-polygon
#: shapefile already in EPSG:32633. The dataset's `attachment/` listing
#: actually goes back to 1936 (1936/1960/1976/1985/1990, then annual from
#: 2012); only 2008 is used here so far, as a pre-surge baseline for
#: `bed.marine_domain` --
#: the older decadal years are left out until independently vetted (different
#: digitization era/method), per the user's explicit caution (2026-09-02).
COASTLINE_URLS: Final[dict[int, str]] = {
    2008: "https://next.api.npolar.no/dataset/21565514-ab3a-432a-ae42-2587bea95a97/attachment/d4c2be0b-cf78-4884-8f24-3dc9d46d20ef/_blob",
    2012: "https://next.api.npolar.no/dataset/21565514-ab3a-432a-ae42-2587bea95a97/attachment/cb09bf5c-7c0a-40e7-9368-b86e7d2bb9f4/_blob",
    2018: "https://next.api.npolar.no/dataset/21565514-ab3a-432a-ae42-2587bea95a97/attachment/dd953513-4743-410b-a51b-461ce16e9e06/_blob",
    2023: "https://next.api.npolar.no/dataset/21565514-ab3a-432a-ae42-2587bea95a97/attachment/e499ee72-65f1-4002-83a3-b8de08649978/_blob",
}


def download_file(url: str, out_path: Path, refresh_download: bool = False) -> Path:
    """Download `url` to `out_path`, skipping it if already present.

    Downloads to a `.tmp`-infixed sibling first and atomically replaces
    `out_path` on success, so an interrupted download never leaves a
    misleading complete-looking file. This is a *source* download, not a
    derived cache, so it takes a plain `refresh_download` flag rather than
    the `ForceRedo` propagation used by derived caches (see 
    the Python-project caching conventions).

    Parameters
    ----------
    url
        Source URL to fetch.
    out_path
        Local destination path. Its parent directory is created if needed.
    refresh_download
        Re-download even if `out_path` already exists.

    Returns
    -------
    Path
        `out_path`, once the file is present locally.
    """
    if out_path.is_file() and not refresh_download:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as outfile:
                outfile.writelines(response.iter_content(chunk_size=8192 * 4))
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return out_path


def get_rgi7(refresh_download: bool = False) -> Path:
    """Download (if needed) the RGI7.0 Svalbard & Jan Mayen release; return its local zip path."""
    return download_file(RGI7_URL, config.RGI7_ZIP, refresh_download=refresh_download)


def get_coastline(year: int, refresh_download: bool = False) -> Path:
    """Download (if needed) the NPI annual coastline for `year`; return its local zip path."""
    if year not in COASTLINE_URLS:
        raise KeyError(
            f"no coastline URL for year {year}; have {sorted(COASTLINE_URLS)}"
        )
    out_path = config.COASTLINE_DIR / f"Coast{year}.zip"
    return download_file(
        COASTLINE_URLS[year], out_path, refresh_download=refresh_download
    )


SVIFT_URL: Final = "https://next.api.npolar.no/dataset/57fd0db4-afbf-4c94-ac1c-191c714f1224/attachment/ec2d911f-97c6-4d78-8602-7539b97470a6/_blob"

#: SVIFT 1.0's `bed.nc`, from the same npolar dataset record as v1.1
#: (`attachment/` listing, not linked from the dataset's rendered page).
#: Kept alongside 1.1 as a second SVIFT version, not as a replacement.
SVIFT_V10_URL: Final = "https://next.api.npolar.no/dataset/57fd0db4-afbf-4c94-ac1c-191c714f1224/attachment/b83c8e9d-e610-4a7c-88b2-ebe76be330b0/_blob"

#: Fill value used inside the SVIFT netCDFs (they carry no `_FillValue`
#: attribute, so `mask_and_scale` cannot pick it up automatically).
SVIFT_NODATA: Final = -9999.0


def _normalize_svift_bed_netcdf(netcdf_path: Path, tif_path: Path) -> Path:
    """Write a north-up GeoTIFF from a raw SVIFT `<ver>_bed.nc`, atomically.

    Both SVIFT versions store rows south-to-north (`y` ascending). GDAL's
    netCDF driver reports a correct north-up transform for them but returns
    the rows in file (south-up) order, so reading either file directly with
    rasterio yields a vertically mirrored bed -- verified for v1.1 by
    rasterizing the RGI7 outlines against SVIFT's valid-data mask: 6.3%
    overlap as read by rasterio, 99.5% once flipped.
    We therefore normalize once here, on extraction, into a plain GeoTIFF
    that every consumer can open without repeating the flip.
    """
    import numpy as np
    import rasterio
    import xarray as xr

    with xr.open_dataset(netcdf_path, mask_and_scale=False) as dataset:
        y_coords = dataset["y"].values
        x_coords = dataset["x"].values
        bed = np.asarray(dataset["bed"].values, dtype=np.float32)

    if y_coords[1] <= y_coords[0]:
        raise ValueError(
            f"{netcdf_path.name} was expected to store y ascending (south-up); "
            "the orientation fix here assumes that and must be rechecked"
        )
    bed = bed[::-1]
    bed[bed == SVIFT_NODATA] = np.nan

    half_pixel = float(x_coords[1] - x_coords[0]) / 2.0
    transform = rasterio.transform.from_origin(
        float(x_coords[0]) - half_pixel,
        float(y_coords[-1]) + half_pixel,
        float(x_coords[1] - x_coords[0]),
        float(y_coords[1] - y_coords[0]),
    )

    tmp_path = tif_path.with_name(f"{tif_path.stem}.tmp.tif")
    try:
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=bed.shape[0],
            width=bed.shape[1],
            count=1,
            dtype="float32",
            crs=config.CRS_EPSG,
            transform=transform,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
        ) as dst:
            dst.write(bed, 1)
        tmp_path.replace(tif_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return tif_path


def get_svift_bed(refresh_download: bool = False) -> Path:
    """Download+extract SVIFT 1.1 and return a north-up GeoTIFF of its bed elevation.

    SVIFT's own grid (checked directly in the netCDF) is already EPSG:32633
    at 100 m resolution, so it defines the target grid for the merged bed
    DEM rather than needing resampling. See `_normalize_svift_bed_netcdf`
    for the orientation fix this applies.

    Parameters
    ----------
    refresh_download
        Re-download, re-extract and rewrite the GeoTIFF even if present.

    Returns
    -------
    Path
        `input/svift/svift_v11/svift_v11_bed.tif`, EPSG:32633, 100 m,
        north-up, nodata NaN. Values are bed elevation in m relative to
        sea level (negative below).
    """
    import tarfile

    tif_path = config.SVIFT_DIR / "svift_v11_bed.tif"
    if tif_path.is_file() and not refresh_download:
        return tif_path

    netcdf_path = config.SVIFT_DIR / "svift_v11_bed.nc"
    if not netcdf_path.is_file() or refresh_download:
        tar_path = download_file(
            SVIFT_URL, config.SVIFT_TAR, refresh_download=refresh_download
        )
        with tarfile.open(tar_path) as tar_file:
            tar_file.extractall(config.SVIFT_DIR.parent, filter="data")

    return _normalize_svift_bed_netcdf(netcdf_path, tif_path)


def get_svift_v10_bed(refresh_download: bool = False) -> Path:
    """Download SVIFT 1.0's bed and return a north-up GeoTIFF (see `get_svift_bed`).

    Unlike v1.1, v1.0 is distributed as a single loose `.nc` (no tar), from
    the same npolar dataset record.

    Parameters
    ----------
    refresh_download
        Re-download and rewrite the GeoTIFF even if present.

    Returns
    -------
    Path
        `input/svift/svift_v10_bed.tif`, EPSG:32633, 100 m, north-up,
        nodata NaN.
    """
    tif_path = config.SVIFT_DIR.parent / "svift_v10_bed.tif"
    if tif_path.is_file() and not refresh_download:
        return tif_path

    netcdf_path = config.SVIFT_DIR.parent / "svift_v10_bed.nc"
    download_file(SVIFT_V10_URL, netcdf_path, refresh_download=refresh_download)

    return _normalize_svift_bed_netcdf(netcdf_path, tif_path)


#: lon_min, lat_min, lon_max, lat_max, generous around Svalbard proper
#: (Jan Mayen is far outside this and is irrelevant here anyway). Used by
#: `get_ibcao_svalbard`, reprojected into IBCAO's native EPSG:3996.
SVALBARD_BBOX_4326: Final[tuple[float, float, float, float]] = (9.0, 75.5, 36.0, 81.5)


#: IBCAO v5.2 (Jakobsson et al. 2024), 100 m Polar Stereographic (EPSG:3996)
#: grid, "no Greenland ice sheet" variant -- the two Greenland-elevation
#: variants differ only over Greenland itself (IBCAO's own readme:
#: "Versions of the data set are made available with and without ice
#: surface elevation information for Greenland"), which never overlaps
#: Svalbard, so the choice is inconsequential here; "no ice sheet" is
#: preferred since we want bathymetry, not ice-surface elevation, wherever
#: the two variants would actually differ. The marine bathymetry source
#: for every `bed.py` model.
#:
#: The depth grid is offered pre-split into a 4x4 tile grid; Svalbard falls
#: entirely within tiles (2,0) and (2,1) (x-column 2 of 0-3 west-to-east,
#: y-row 0-1 of 0-3 south-to-north), each 1455 km square. We use that split
#: (rather than the single 58200x58200 whole-Arctic grid) specifically to
#: avoid transferring unrelated-region strips: both are stored as
#: single-row TIFF strips (not tiled), so any windowed read still pulls
#: full-width strips for every row in the window -- splitting into
#: 1/16th-sized tiles keeps "full width" 4x narrower than reading from the
#: whole-Arctic grid would. The *zipped* copies are DEFLATE-compressed
#: (confirmed via zip central directory: compress_type=8), which breaks
#: efficient HTTP range reads (GDAL must decompress from the start of the
#: stream); the plain (unzipped) `.tiff` siblings CEDA also publishes
#: support HTTP range requests directly and are used here instead.
#: The TID grid has no tiled split, so it's windowed-read from the single
#: whole-Arctic grid -- still only ~500 MB for the Svalbard row range
#: (uint8, so no worse than the depth tiles despite the 4x wider strips).
IBCAO_DEPTH_TILE_URLS: Final[dict[tuple[int, int], str]] = {
    (2, 0): (
        "/vsicurl/https://dap.ceda.ac.uk/bodc/gebco/ibcao/ibcao_v5.2/"
        "no_greenland_ice_sheet_elevation_data/100mx100m_grid_cell_spacing/"
        "16_tiles/ibcao_v5_2_2026_depth_part_2_0.tiff"
    ),
    (2, 1): (
        "/vsicurl/https://dap.ceda.ac.uk/bodc/gebco/ibcao/ibcao_v5.2/"
        "no_greenland_ice_sheet_elevation_data/100mx100m_grid_cell_spacing/"
        "16_tiles/ibcao_v5_2_2026_depth_part_2_1.tiff"
    ),
}
IBCAO_TID_URL: Final = (
    "/vsicurl/https://dap.ceda.ac.uk/bodc/gebco/ibcao/ibcao_v5.2/"
    "100mx100m_data_type_identifier_grid/ibcao_v5_2_2026_tid_100m.tiff"
)

#: GDAL/curl tuning for the single-row-strip IBCAO grids: the 16 KB default
#: HTTP chunk size means one request per strip, which is impractically slow
#: (a few thousand round trips) for a multi-thousand-row window; a 1 MB
#: chunk size cuts that by ~65x for these float32/uint8, 14550-58200-wide
#: strips. Confirmed empirically: the tuned settings read the full Svalbard
#: window (~66M pixels across the two depth tiles) in ~195s total, vs. the
#: untuned default not completing within several minutes.
_IBCAO_GDAL_ENV: Final[dict[str, str]] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_CHUNK_SIZE": "1048576",
    "VSI_CACHE": "YES",
    "VSI_CACHE_SIZE": "200000000",
}


def get_ibcao_svalbard(
    kind: Literal["depth", "tid"], refresh_download: bool = False
) -> Path:
    """Crop the IBCAO v5.2 bathymetry or TID grid to Svalbard; return the local GeoTIFF path.

    The bathymetry source for every `bed.py` model.

    Parameters
    ----------
    kind
        `"depth"` or `"tid"`.
    refresh_download
        Re-crop even if the local file already exists.

    Returns
    -------
    Path
        `input/ibcao/ibcao_v5_2_svalbard_<kind>.tif`, EPSG:3996 (IBCAO's
        native Polar Stereographic projection -- reprojected to EPSG:32633
        only later, in `bed.py`).
    """
    import rasterio
    import rasterio.warp
    import rasterio.windows

    out_path = config.IBCAO_DIR / f"ibcao_v5_2_svalbard_{kind}.tif"
    if out_path.is_file() and not refresh_download:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Svalbard bbox in IBCAO's native EPSG:3996, densified before
    # reprojecting (polar stereographic distorts a lon/lat box into a
    # non-rectangular shape) and padded by 5 km.
    xmin, ymin, xmax, ymax = rasterio.warp.transform_bounds(
        "EPSG:4326", "EPSG:3996", *SVALBARD_BBOX_4326, densify_pts=21
    )
    xmin, ymin, xmax, ymax = xmin - 5000, ymin - 5000, xmax + 5000, ymax + 5000

    with rasterio.Env(**_IBCAO_GDAL_ENV):
        if kind == "tid":
            with rasterio.open(IBCAO_TID_URL) as src:
                window = (
                    rasterio.windows.from_bounds(
                        xmin, ymin, xmax, ymax, transform=src.transform
                    )
                    .round_lengths()
                    .round_offsets()
                )
                data = src.read(1, window=window)
                transform = rasterio.windows.transform(window, src.transform)
                profile = src.profile.copy()
        elif kind == "depth":
            tiles: dict[tuple[int, int], tuple[np.ndarray, affine.Affine]] = {}
            for key, url in IBCAO_DEPTH_TILE_URLS.items():
                with rasterio.open(url) as src:
                    window = (
                        rasterio.windows.from_bounds(
                            max(xmin, src.bounds.left),
                            max(ymin, src.bounds.bottom),
                            min(xmax, src.bounds.right),
                            min(ymax, src.bounds.top),
                            transform=src.transform,
                        )
                        .round_lengths()
                        .round_offsets()
                    )
                    tiles[key] = (
                        src.read(1, window=window),
                        rasterio.windows.transform(window, src.transform),
                    )
                    profile = src.profile.copy()
            # Both covering tiles share the same x-range/columns (identical
            # left bound and pixel grid) and are vertically adjacent
            # (row-index 0 south of row-index 1), so a plain vertical stack
            # (north tile's rows on top, matching its own window transform)
            # reproduces the merged grid without needing a general mosaic
            # routine. `north_data.shape[1] == south_data.shape[1]` is
            # asserted rather than silently mismatched/broadcast.
            south_data, _ = tiles[(2, 0)]
            north_data, north_transform = tiles[(2, 1)]
            if north_data.shape[1] != south_data.shape[1]:
                raise ValueError(
                    "IBCAO depth tile windows have mismatched widths: "
                    f"north={north_data.shape[1]}, south={south_data.shape[1]}"
                )
            data = np.vstack([north_data, south_data])
            transform = north_transform
        else:
            raise KeyError(f"kind must be 'depth' or 'tid', got {kind!r}")

    profile.update(
        height=data.shape[0],
        width=data.shape[1],
        transform=transform,
        driver="GTiff",
        compress="deflate",
    )

    tmp_path = out_path.with_name(f"{out_path.stem}.tmp.tif")
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(data, 1)
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return out_path


#: IceBoost v2.0 (Maffezzoli et al.), regional RGI7 mosaics. Two UTM zones
#: cover Svalbard: 32633 (most of it, our project CRS) and 32635 (the
#: eastern islands -- Nordaustlandet, Kvitøya -- fall outside the 32633
#: tile's own extent). Bands: thickness, thickness_err, jensen_gap, h_wgs84
#: (TanDEM-X surface, ellipsoidal), n_geoid (EIGEN-6C4 geoid height); bed
#: elevation is `(h_wgs84 - n_geoid) - thickness`, computed in `bed.py`.
MAFFEZZOLI_TILE_URLS: Final[dict[int, str]] = {
    32633: (
        "/vsizip//vsicurl/https://zenodo.org/records/20463551/files/"
        "IceBoostv20_complexes_RGI_7.zip/complex/rgi7/"
        "iceboost_20251009_rgi7_v70G_epsg_32633.tif"
    ),
    32635: (
        "/vsizip//vsicurl/https://zenodo.org/records/20463551/files/"
        "IceBoostv20_complexes_RGI_7.zip/complex/rgi7/"
        "iceboost_20251009_rgi7_v70G_epsg_32635.tif"
    ),
}


def get_maffezzoli_tile(epsg: int, refresh_download: bool = False) -> Path:
    """Download (if needed) one Maffezzoli IceBoost v2.0 RGI7 UTM tile; return its local path.

    The upstream file lives inside a 1.4 GB zip
    (`IceBoostv20_complexes_RGI_7.zip`) alongside every other RGI7 UTM zone;
    rather than downloading that whole archive, this reads the one tile we
    need directly off GDAL's `/vsizip//vsicurl/` virtual filesystem (window
    reads over HTTP, no local copy of the archive) and writes just that
    tile locally -- the same deliberate exception to full-download-then-use
    used for IBCAO.

    Parameters
    ----------
    epsg
        One of `MAFFEZZOLI_TILE_URLS` (32633 or 32635).
    refresh_download
        Re-fetch even if the local file already exists.

    Returns
    -------
    Path
        `input/maffezzoli/iceboost_rgi7_epsg_<epsg>.tif`, 5 bands (see
        module-level comment), native CRS/resolution (100 m).
    """
    import rasterio

    if epsg not in MAFFEZZOLI_TILE_URLS:
        raise KeyError(
            f"epsg must be one of {sorted(MAFFEZZOLI_TILE_URLS)}, got {epsg!r}"
        )

    out_path = config.MAFFEZZOLI_DIR / f"iceboost_rgi7_epsg_{epsg}.tif"
    if out_path.is_file() and not refresh_download:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(MAFFEZZOLI_TILE_URLS[epsg]) as src:
        profile = src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(driver="GTiff", compress="deflate", tiled=True)
        data = src.read()
        descriptions = src.descriptions

        tmp_path = out_path.with_name(f"{out_path.stem}.tmp.tif")
        try:
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(data)
                for band, description in enumerate(descriptions, start=1):
                    if description:
                        dst.set_band_description(band, description)
            tmp_path.replace(out_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    return out_path


#: Millan et al. (2022) ice thickness, already EPSG:32633/50 m. RGI-7.1
#: covers most of Svalbard, RGI-7.2 the eastern extension (Nordaustlandet
#: side); together they span the same footprint as the two Maffezzoli
#: tiles combined. No bed layer -- combined with `get_npi_dem()` in
#: `bed.py` (`bed = dem - thickness`).
MILLAN_THICKNESS_URLS: Final[dict[str, str]] = {
    "7.1": "https://cluster.klima.uni-bremen.de/~oggm/velocities/millan22/thickness/RGI-7/THICKNESS_RGI-7.1_2021July09.tif",
    "7.2": "https://cluster.klima.uni-bremen.de/~oggm/velocities/millan22/thickness/RGI-7/THICKNESS_RGI-7.2_2021July09.tif",
}


def get_millan_thickness(tile: str, refresh_download: bool = False) -> Path:
    """Download (if needed) one Millan et al. (2022) RGI-7 thickness tile; return its local path.

    Parameters
    ----------
    tile
        One of `MILLAN_THICKNESS_URLS` (`"7.1"` or `"7.2"`).
    refresh_download
        Re-download even if the local file already exists.

    Returns
    -------
    Path
        `input/millan/THICKNESS_RGI-<tile>.tif`, EPSG:32633, 50 m, nodata 0.
    """
    if tile not in MILLAN_THICKNESS_URLS:
        raise KeyError(
            f"tile must be one of {sorted(MILLAN_THICKNESS_URLS)}, got {tile!r}"
        )

    out_path = config.MILLAN_DIR / f"THICKNESS_RGI-{tile}.tif"
    return download_file(
        MILLAN_THICKNESS_URLS[tile], out_path, refresh_download=refresh_download
    )


NPI_DEM_URL: Final = "https://next.api.npolar.no/dataset/dce53a47-c726-4845-85c3-a65b46fe2fea/attachment/838be1e3-58ef-4338-b80c-dec86ba07bd2/_blob"


def get_npi_dem(refresh_download: bool = False) -> Path:
    """Download+extract the NPI 50 m DTM; return the local path to its GeoTIFF.

    Needed to convert Millan et al.'s ice-thickness-only product to a bed
    elevation (`bed = dem - thickness` in `bed.py`). The zip is served with
    on-the-fly gzip content-encoding, which breaks HTTP range reads against
    it, so (unlike IBCAO/Maffezzoli) this is a plain full download.

    Parameters
    ----------
    refresh_download
        Re-download/re-extract even if the local GeoTIFF already exists.

    Returns
    -------
    Path
        The extracted DTM GeoTIFF under `input/dem/NP_S0_DTM50/`.
    """
    import zipfile

    tif_candidates = (
        list(config.DEM_DIR.rglob("*.tif")) if config.DEM_DIR.is_dir() else []
    )
    if tif_candidates and not refresh_download:
        return tif_candidates[0]

    zip_path = download_file(
        NPI_DEM_URL, config.DEM_ZIP, refresh_download=refresh_download
    )
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(config.DEM_DIR)

    tif_candidates = list(config.DEM_DIR.rglob("*.tif"))
    if not tif_candidates:
        raise FileNotFoundError(
            f"no .tif found in {NPI_DEM_URL} after extracting to {config.DEM_DIR}"
        )
    return tif_candidates[0]


VANPELT_FRANK_BED_URL: Final = (
    "https://zenodo.org/records/11239460/files/Bed_map.tif?download=1"
)


def get_vanpelt_frank_bed(refresh_download: bool = False) -> Path:
    """Download (if needed) van Pelt & Frank (2025)'s Svalbard bed map; return its local path.

    Parameters
    ----------
    refresh_download
        Re-download even if the local file already exists.

    Returns
    -------
    Path
        `input/vanpelt/Bed_map.tif`, EPSG:25833, 100 m, bed elevation in m
        a.s.l. Reprojected onto the low-resolution (500 m)
        target grid in `bed.py`, not here.
    """
    out_path = config.VANPELT_DIR / "Bed_map.tif"
    return download_file(
        VANPELT_FRANK_BED_URL, out_path, refresh_download=refresh_download
    )


VANPELT_FRANK_MASK_URL: Final = (
    "https://zenodo.org/records/11239460/files/Mask.tif?download=1"
)


def get_vanpelt_frank_mask(refresh_download: bool = False) -> Path:
    """Download (if needed) van Pelt & Frank (2025)'s model-domain mask; return its local path.

    Values 0-3: non-glacier (0, unmodelled -- `Bed_map.tif` is a 0 m fill
    there, not a real bed value), PISM-modelled (1), IGM-modelled (2),
    surging-glacier (3). Used in `bed.py` to mask `Bed_map.tif` to cells the
    model actually ran on (see AGENTS.md).

    Parameters
    ----------
    refresh_download
        Re-download even if the local file already exists.

    Returns
    -------
    Path
        `input/vanpelt/Mask.tif`, same grid as `Bed_map.tif`.
    """
    out_path = config.VANPELT_DIR / "Mask.tif"
    return download_file(
        VANPELT_FRANK_MASK_URL, out_path, refresh_download=refresh_download
    )


class _HTTPRangeReader:
    """Minimal seekable file-like object doing one HTTP range request per read.

    Used only for `get_frank_topg`'s figshare download: figshare's
    `ndownloader` redirects to a pre-signed S3 URL that expires in ~10
    seconds, too short to reuse across the many small reads a zip central-
    directory listing needs. Re-resolving `url` (following the redirect
    fresh) on every read sidesteps that, at the cost of one extra request
    per read -- acceptable here since we only ever read a handful of ranges
    (the outer zip's central directory, then one inner-zip entry).
    """

    def __init__(self, url: str, session: requests.Session):
        self.url = url
        self.session = session
        self._pos = 0
        self._size = self._fetch_size()

    def _fetch_size(self) -> int:
        response = self.session.get(
            self.url, headers={"Range": "bytes=0-0"}, timeout=30
        )
        return int(response.headers["Content-Range"].split("/")[-1])

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        end = (self._size if n is None or n < 0 else self._pos + n) - 1
        end = min(end, self._size - 1)
        if self._pos > end:
            return b""
        response = self.session.get(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=60
        )
        data = response.content
        self._pos += len(data)
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


#: Frank et al. (in review/2026) subglacial topography ("Topg"), from the
#: figshare record for Frank & van Pelt's future-lakes study. One 704 MB
#: zip-of-zips for the whole world; `RGI-07_topg.zip` (the Svalbard entry,
#: ~22 MB, stored uncompressed inside the outer zip) is extracted directly
#: over HTTP range reads via `_HTTPRangeReader` rather than downloading the
#: full 704 MB. Per-glacier tiles inside use RGI **6.0** IDs (the dataset
#: predates RGI7); irrelevant here since we attribute by geometry, not ID.
FRANK_TOPG_URL: Final = "https://ndownloader.figshare.com/files/57288194"
FRANK_TOPG_INNER_ENTRY: Final = "RGI-07_topg.zip"


def get_frank_topg(refresh_download: bool = False) -> Path:
    """Fetch Frank et al.'s Svalbard (RGI-07) subglacial-topography zip; return its local path.

    Parameters
    ----------
    refresh_download
        Re-fetch even if the local file already exists.

    Returns
    -------
    Path
        `input/frank/RGI-07_topg.zip`, containing one GeoTIFF per glacier
        complex (`topg_final/RGI60-07.<id>_topg.tif`), each already
        EPSG:32633/100 m, nodata 0 (see `bed.py` for the mosaic step).
    """
    import zipfile

    out_path = config.FRANK_DIR / FRANK_TOPG_INNER_ENTRY
    if out_path.is_file() and not refresh_download:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        reader = _HTTPRangeReader(FRANK_TOPG_URL, session)
        with zipfile.ZipFile(reader) as outer_zip:
            info = outer_zip.getinfo(FRANK_TOPG_INNER_ENTRY)
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(
                    f"expected {FRANK_TOPG_INNER_ENTRY} to be stored uncompressed "
                    f"inside the outer zip, got compress_type={info.compress_type}; "
                    "the streamed single-range read below assumes STORED"
                )
            reader.seek(info.header_offset)
            local_header = reader.read(30)
            name_len = int.from_bytes(local_header[26:28], "little")
            extra_len = int.from_bytes(local_header[28:30], "little")
            reader.seek(info.header_offset + 30 + name_len + extra_len)
            data = reader.read(info.compress_size)

        tmp_path = out_path.with_name(f"{out_path.stem}.tmp.zip")
        try:
            tmp_path.write_bytes(data)
            tmp_path.replace(out_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    return out_path
