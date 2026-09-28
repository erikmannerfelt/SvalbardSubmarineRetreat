# Tests

```
python -m pytest tests/ -q
```

**These are pure unit tests.** They construct their own synthetic rasters and
geometries and touch neither `input/` nor `cache/`, so they run in a few
seconds and need no downloads.

## What is covered, and why

- `test_harmonic_fill.py` — the Laplace gap fill supplies most of the bed
  value at the calving fronts, so its correctness is load-bearing. Pins
  recovery of known harmonic fields, convergence, the sign guarantee the
  marine solve depends on, and the always-sea clamp's motivation.
- `test_extraction.py` — the per-glacier clip/difference that builds the
  change polygons: attribution is immediate because the outlines are
  disjoint, ice-free rock cancels in the difference, and the geometry
  hygiene that removes spikes and sub-millimetre dust.
- `test_domain_helpers.py` — the small pure helpers in `bed.py` and
  `outlines.py`: cache paths, provenance codes, the TID whitelist, the
  harmonic fill on a constant field, the tidewater flag including the
  Scheelebreen override, and the known outline-correction active-year
  exceptions.
- `test_volume.py` — the per-polygon submarine depth sampling: the
  `max(0, -bed)` depth convention, the area-weighted volume integral, recovery
  of a thin sliver that misses every cell centre, exclusion and counting of
  cells without a value, and the interpolation share.
