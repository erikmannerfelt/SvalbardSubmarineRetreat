# Submarine retreat for tidewater glaciers in Svalbard

Two deliverables for estimating submarine volume change of Svalbard
tidewater glaciers (RGI7 tidewater plus Scheelebreen):

1. **Change polygons** -- each tidewater glacier's retreat and advance for
   2012-2018, 2018-2023 and 2012-2023, attributed to its RGI7 ID
   (`cache/change_polygons_by_rgi.arrow`).
2. **Bed surfaces** -- a gap-free 100 m bed-elevation raster for each of six
   subglacial bed/ice-thickness products (`products/bed/<model>_bed_100m.tif`).

Multiplying polygon area by the mean bed depth below sea level over that
polygon gives the submarine volume change; that sampling step is downstream
of this repo's outputs.

`AGENTS.md` records the durable domain rules behind the non-obvious choices,
in particular the Nathorstbreen Glacier System (NGS) surge fixes.

## Pipeline

1. **Corrected outlines** (`src/outlines.py`) -- RGI7 region 7, Jan Mayen
   dropped, the Scheelebreen tidewater override, the manual geometry
   corrections in `shapes/outline_corrections.geojson`, and a tidewater flag
   (`term_type == 1` or `zmin_m <= 25`).
2. **Change polygons** (`src/extraction.py`) -- each year's NPI coastline is
   clipped to each glacier's own outline to get that glacier's extent, and
   consecutive extents are differenced. Attribution is immediate, no area can
   be counted twice, and ice-free rock cancels. A 150 m ring outside the
   outlines recovers change the outlines do not cover.
3. **Bed surfaces** (`src/bed.py`) -- for each source, four provenance layers
   are merged on SVIFT's native 100 m EPSG:32633 grid: IBCAO bathymetry at
   sea, the source's own inversion under ice and elsewhere at sea, the NPI
   DEM on ice-free land, and a harmonic (Laplace) fill for whatever is left.
   Marine and land/glacier gaps are filled by separate solves so a land cell
   can never bleed into a fjord. The six models are `svift_v11`, `svift_v10`,
   `maffezzoli`, `millan`, `frank` and `vanpelt` (`bed.BED_MODELS`).

## Running

```python
from src import bed, extraction

extraction.make_rgi_attributed_changes()   # writes the change-polygon output
bed.export_all_products()                  # builds + writes every bed raster
```

Derived caches take a `force_redo` flag: `False` reuses a cache, `"self"`
recomputes the artifact from cached upstream outputs, `"all"` also recomputes
the dependencies. Source downloads use a separate `refresh_download` flag.
`products/bed/*.tif` are copies of `cache/bed_<model>.tif`; re-run
`bed.export_all_products()` after any rebuild, and delete any stale GDAL
`.aux.xml` sidecars a viewer may have written.

## Products

`cache/change_polygons_by_rgi.arrow` -- one row per (glacier, interval, change
type), columns `rgi_id`, `glac_name`, `year_from`, `year_to`, `change_type`,
`area_m2`, `geometry`, EPSG:32633.

`products/bed/<model>_bed_100m.tif` -- three bands, EPSG:32633, 100 m, NaN
nodata: (1) bed elevation in m relative to sea level, negative below; (2)
distance in m to the nearest directly measured cell; (3) a provenance code
(`bed.SOURCE_CODE_LABELS`: inversion / bathymetry / DEM / interpolation).

`products/bed/bed_spread_100m.tif` -- per-pixel ensemble spread over
`bed.ENSEMBLE_MODELS` (the five estimates; `svift_v10` is excluded as a
second SVIFT version): (1) standard deviation of bed elevation, (2) the mean,
(3) the number of models with a value. Most cells have zero spread because
all models share the IBCAO bathymetry and NPI DEM, so read band 1 only where
a subglacial inversion is doing the work.

## Notes

**NGS and other pre-2012 surges.** `bed.marine_domain` asks whether a cell was
ever open water in 2008 or 2012/2018/2023, not just the three study years: the
Nathorstbreen Glacier System surged over open water around 2008, and that
still-ice-covered area would otherwise be treated as ordinary land. Frank's
model also still sees the post-surge fjord as ice-free land, so
`shapes/ngs_frank_override.geojson` discards Frank there by hand. See
`AGENTS.md`.

**Interpolation share.** Much of the bed value at the calving fronts comes
from this repo's harmonic fill rather than any source model, most for the
SVIFT pair. The `source_code` band lets any downstream estimate report how
much of its depth rests on measured data.

**Outline debt.** Some RGI7 outlines predate a surge and do not cover their
own glacier's change; `shapes/OUTLINE_PROBLEMS.md` ranks the affected
glaciers and should be worked through before the outlines are final.

## Data sources

Every dataset has a `get_<source>()` in `src/sources.py` that downloads to
`input/` on demand, so `input/` is reproducible from scratch.

- **RGI7.0** region 07 (Svalbard & Jan Mayen):
  `https://cluster.klima.uni-bremen.de/~fmaussion/fmaussion/misc/rgi7_data/rgi70_official/`
- **NPI annual coastlines** (2012/2018/2023, plus 2008 for the marine domain):
  `https://data.npolar.no/dataset/21565514-ab3a-432a-ae42-2587bea95a97`
- **IBCAO v5.2** 100 m bathymetry + TID grid, "no Greenland ice sheet"
  variant, pre-split tiles: `https://www.gebco.net/data-products/gridded-bathymetry-data/arctic-ocean`.
  Only TID codes 10-17 (direct measurement) are kept.
- **SVIFT 1.1 / 1.0** bed: `https://data.npolar.no/dataset/57fd0db4-afbf-4c94-ac1c-191c714f1224`
- **Maffezzoli et al. IceBoost v2.0** (ice thickness + surface + geoid):
  `https://zenodo.org/records/20463551`
- **Millan et al. 2022** (ice thickness only, needs the DEM):
  `https://cluster.klima.uni-bremen.de/~oggm/velocities/millan22/thickness/RGI-7/`
- **Frank et al.** subglacial topography: `https://doi.org/10.6084/m9.figshare.29940932`
- **van Pelt & Frank 2025** bed + model mask: `https://zenodo.org/records/11239460`
- **NPI 50 m DTM**: `https://data.npolar.no/dataset/dce53a47-c726-4845-85c3-a65b46fe2fea`

## Related repos

- Union masks with glacier attribution from RGI7/coastlines/outline
  corrections: `https://github.com/erikmannerfelt/ad_svalbard_post`
- Related glacier volume processing:
  `https://github.com/joh-brunner/glacier_area_volume`
- Downloaders for Millan, van Pelt and others:
  `https://github.com/erikmannerfelt/svalbard_radar`
