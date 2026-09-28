# AGENTS.md

Working memory for `submarine_retreat_svalbard`. The global container/Python
conventions from `~/.config/opencode/AGENTS.md` (src/ layout, typed +
frozen-dataclass style, `force_redo: ForceRedo` cache pattern, Feather/Arrow
for cached vector data, etc.) apply here. This file records only the durable
domain reasoning behind the non-obvious choices. See `README.md` for the
pipeline and outputs.

## Project scope

The outputs (see README): the per-glacier retreat/advance change polygons, a
merged gap-free 100 m bed raster per bed/thickness product, the per-pixel
ensemble spread over the five estimates, and the per-polygon submarine volume
changes. The bed rasters are built by `src/bed.py`; `src/volume.py` samples
them into the volume product.

## Bed construction rules, in the order they apply

Four constraints shape every merged bed. Each was added against a measured
failure, and the later ones compose with the earlier ones rather than
replacing them.

1. **Permanently-open water cannot have a bed above sea level.** Cells the
   coastlines call sea in *every* observed year (`bed.always_sea_domain`) keep
   the `elevation <= 0` gate. Interpolated cells there are clipped to 0 as
   well, since admitting positive knowns elsewhere costs the maximum
   principle.
2. **Inside the coastline's reach an inversion may sit above sea level.** The
   cell was land or ice at some epoch, so its bed can be either sign. Only
   inversion values are exempt; bathymetry and DEM positives in the marine
   domain are the contamination the gate exists for.
3. **Unless the bed coincides with the surface** (`SURFACE_MATCH_TOLERANCE_M`
   = 5 m), which is a zero-thickness assertion and contradicts the coastline.
4. **No bed may exceed the NPI DEM surface** (`CLAMP_BED_TO_SURFACE`), applied
   to the sources and again to the filled field. Clamped rather than
   discarded, which makes it a clamp on land and, via rule 3, a rejection at
   sea.

Plus one per-model exception: `shapes/ngs_frank_override.geojson` discards
`frank` over the Nathorstbreen surge footprint, where its ice mask still treats
the post-surge fjord as ice-free land. Hand-drawn because no geometric proxy
worked. See `bed.frank_override_zone`.

The marine and land/glacier gaps are interpolated as two *separate* harmonic
(Laplace) solves, so a land cell can never act as a Dirichlet neighbour for a
marine gap. With every boundary value the marine solve sees held to <= 0, the
maximum principle guarantees every interpolated marine cell is too.

## NGS pre-2012 surge blind spot (2008 coastline)

`bed.marine_domain` tests "was this ever open water" over
`config.MARINE_DOMAIN_YEARS = (2008, 2012, 2018, 2023)`, not just the three
study years. The Nathorstbreen Glacier System surged over open water around
2008; that area is still ice-covered through 2012-2023, so a study-year-only
test misclassifies it as land and the fill gives it a positive bed. NPI's
coastline archive goes back to 1936 but only 2008 is used, pending independent
vetting of the older digitization.

The same event is why `SURFACE_MATCH_TOLERANCE_M` exists: Frank's mask says the
surge area is ice-free land, so its "bed" is the stale surge surface, ~166 m
above sea level in a fjord whose floor is about -350 m. Rule 4 clamps it to
the DEM (still positive), and rule 3 then screens it inside the marine change
zone.

The change polys themselves always use `config.COASTLINE_YEARS` only
(2012/2018/2023); the 2008 coastline widens the marine domain, never the
retreat/advance output.

## Change polygons are attributed by construction

`extraction.py` clips each year's coastline to each glacier's RGI7 outline
first, then differences consecutive extents. Because the outlines are
disjoint, a change polygon belongs to exactly one glacier and no area can be
counted twice -- this replaced an earlier whole-Svalbard difference plus a
buffered-outline attribution that double-counted confluent fronts (first seen
at the NGS). A 150 m ring outside the outlines recovers
change the outlines miss; each connected piece is assigned whole to its
nearest glacier (validated: whole-piece assignment misplaces 0.0087% of
attributed change area).

Geometry hygiene matters: the difference chain leaves zero-width spikes that
are valid but set the bounding box `bed._fill_aoi_geometry` derives the bed
crop from. `extraction._clean_geometry` snaps to a 1 cm grid and drops
sub-millimetre parts; one measured spike ran 1099 m out and was extending the
bed domain ~1 km.

## Submarine volume change is a coverage-weighted depth integral

`volume.py` samples each bed raster over each change polygon. The depth at a
cell is `max(0, -bed)`: a bed at or above sea level contributes zero submarine
depth, so the volume is a true below-sea-level volume rather than a signed
elevation integral. Each cell's depth is weighted by the fraction of that 100 m
cell the polygon covers (`volume._cell_overlaps`), and the volume change is
the weighted integral; the reported mean depth is that integral over the
covered area.

The weighting is deliberate. The change polygons run along calving fronts, so
many are sub-pixel or thin ribbons: a cell-centre test silently drops 134 of
806 polygons (0.033% of the change area), while an unweighted touched-cell
count would include cells a polygon barely clips and inflate the result. It
also means a partly covered polygon gets a volume over the area that was
actually sampled, not one extrapolated from the full polygon area.

Cells where a model's raster has no value (an unconstrained interpolation
component) carry no weight and are counted in `<model>_valid_cells` (against
`n_cells`, the cells the polygon overlaps). The per-model interpolated fraction
is carried so a downstream estimate can report how much of a depth rests on
measured data. `svift_v10` is sampled as a sixth product but excluded from the
ensemble, since it is a second version of the same SVIFT family.

## Naming

SVIFT 1.1 is the model key `svift_v11` and SVIFT 1.0 is `svift_v10` everywhere
(a bare `svift` was ambiguous and was renamed). Both versions are kept as
separate published products.

## Caching

Derived artifacts are `.arrow`/GeoTIFF caches in `cache/`, each with the
`force_redo: config.ForceRedo` pattern (`False` reuse, `"self"` recompute from
cached upstream, `"all"` propagate to dependencies). Source downloads take a
separate `refresh_download` flag. Caches are written atomically via a
same-directory temporary path and re-read after writing. `products/bed/*.tif`
are copies of `cache/bed_<model>.tif` and `products/change_polygons_by_rgi.arrow`
is a copy of `cache/change_polygons_by_rgi.arrow`, each republished with
`bed.export_all_products()` and `extraction.export_change_polygons_product()`
after a rebuild; stale GDAL `.aux.xml` sidecars should be deleted. The two
directly-computed products, `products/bed/bed_spread_100m.tif` and
`products/submarine_volume_change.arrow`, are written atomically straight to
`products/` and re-read after writing.

## Outline debt

Some RGI7 outlines predate a surge and do not cover their own glacier's
change; the outlines should not be treated as final until that debt is worked
through. The affected glaciers are not yet enumerated in-repo.
