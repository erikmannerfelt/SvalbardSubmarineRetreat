"""Unit tests for the direct per-glacier extraction (`src/extraction.py`)."""

from __future__ import annotations

import pytest
import shapely

from src import extraction


def _tree_for(geoms):
    parts = shapely.get_parts(shapely.union_all(geoms))
    return parts, shapely.STRtree(parts)


def test_clip_returns_the_overlap_with_land():
    land = [shapely.box(0, 0, 10, 10), shapely.box(50, 50, 60, 60)]
    parts, tree = _tree_for(land)
    outline = shapely.box(5, 5, 15, 15)

    clipped = extraction._clip_to(outline, parts, tree)

    assert clipped.area == 25.0  # only the 5x5 overlap with the first land part
    assert clipped.within(outline)


def test_clip_ignores_land_outside_the_outline():
    """The distant land part must not leak in through the spatial index."""
    land = [shapely.box(0, 0, 10, 10), shapely.box(1000, 1000, 1010, 1010)]
    parts, tree = _tree_for(land)

    clipped = extraction._clip_to(shapely.box(0, 0, 10, 10), parts, tree)

    assert clipped.area == 100.0


def test_clip_returns_empty_when_the_outline_is_all_sea():
    land = [shapely.box(0, 0, 10, 10)]
    parts, tree = _tree_for(land)

    clipped = extraction._clip_to(shapely.box(100, 100, 110, 110), parts, tree)

    assert clipped.is_empty


def test_clip_of_a_fully_ice_covered_outline_is_the_outline():
    land = [shapely.box(-100, -100, 100, 100)]
    parts, tree = _tree_for(land)
    outline = shapely.box(0, 0, 10, 10)

    clipped = extraction._clip_to(outline, parts, tree)

    assert clipped.equals(outline)


def test_retreat_and_advance_are_complementary_differences():
    """The invariant `make_rgi_attributed_changes` relies on, stated explicitly.

    Retreat and advance are the two one-sided differences of the same pair of
    extents, so they cannot overlap, and each is confined to the outline both
    extents were clipped to.
    """
    outline = shapely.box(0, 0, 10, 10)
    early = shapely.box(0, 0, 10, 8)  # ice reaching further down-fjord
    late = shapely.box(0, 2, 10, 10)  # retreated at one end, advanced at the other

    retreat = shapely.difference(early, late)
    advance = shapely.difference(late, early)

    assert retreat.area == 20.0
    assert advance.area == 20.0
    assert retreat.intersection(advance).area == 0.0
    assert retreat.within(outline) and advance.within(outline)


def test_ice_free_rock_inside_the_outline_cancels_in_the_difference():
    """Land that is land in both years must produce no change signal.

    An RGI outline can contain nunataks and lateral rock, which are land in
    every coastline year. They enter both extents identically and therefore
    drop out of the difference, so no filtering for them is needed.
    """
    rock = shapely.box(2, 2, 4, 4)
    early = shapely.union(shapely.box(0, 0, 10, 8), rock)
    late = shapely.union(shapely.box(0, 2, 10, 10), rock)

    retreat = shapely.difference(early, late)
    advance = shapely.difference(late, early)

    assert not retreat.intersects(rock.buffer(-0.01))
    assert not advance.intersects(rock.buffer(-0.01))


def test_output_columns_cover_the_change_polygon_contract():
    required = {"rgi_id", "year_from", "year_to", "change_type", "area_m2", "geometry"}
    assert required.issubset(set(extraction.OUTPUT_COLUMNS))


def test_change_polygons_are_a_named_output():
    assert extraction.CHANGE_POLYGONS_PATH.name == "change_polygons_by_rgi.arrow"


def test_residual_reach_is_150_m():
    """The ring width that captures 99.99% of attributed change area."""
    assert extraction.RESIDUAL_REACH_M == 150.0


def test_output_columns_carry_the_glacier_identity():
    """Attribution is part of the output, not a later join."""
    assert extraction.OUTPUT_COLUMNS[0] == "rgi_id"
    assert "glac_name" in extraction.OUTPUT_COLUMNS


def test_clean_geometry_removes_a_collinear_spike():
    """A ring that runs out and back along the same line adds no area.

    Seen in the real output (2026-09-17): a 1099 m excursion exactly
    collinear with its neighbours, contributing 0 m^2 but visible in any
    map and, worse, setting the bounding box `bed._fill_aoi_geometry`
    derives the bed crop from.
    """
    # The real spike was a sliver, not a retraced segment: its apex sat
    # 1099 m out and collinear with its neighbours to within floating-point
    # noise (triangle area 2.2e-8 m^2). That is why it stayed valid.
    spiked = shapely.Polygon(
        [(0, 0), (100, 0), (100, 100), (60, 100), (1100, 100.0005), (59, 100), (0, 100)]
    )
    assert spiked.is_valid  # this is why buffer(0) cannot fix it
    assert shapely.get_coordinates(spiked)[:, 0].max() == 1100

    cleaned = extraction._clean_geometry(spiked)

    assert shapely.get_coordinates(cleaned)[:, 0].max() == 100
    assert cleaned.area == pytest.approx(10000.0, rel=1e-6)


def test_clean_geometry_drops_dust_but_keeps_real_pieces():
    real = shapely.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    dust = shapely.Polygon([(500, 500), (500.01, 500), (500.01, 500.01), (500, 500.01)])
    keeper = shapely.Polygon([(700, 700), (703, 700), (703, 703), (700, 703)])

    cleaned = extraction._clean_geometry(shapely.union_all([real, dust, keeper]))

    assert len(shapely.get_parts(cleaned)) == 2
    assert cleaned.area == pytest.approx(real.area + keeper.area, rel=1e-6)


def test_clean_geometry_drops_non_polygonal_components():
    """Some rows came back as GeometryCollections carrying stray LineStrings."""
    collection = shapely.GeometryCollection(
        [
            shapely.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            shapely.LineString([(200, 200), (300, 300)]),
        ]
    )

    cleaned = extraction._clean_geometry(collection)

    assert cleaned.geom_type == "Polygon"
    assert cleaned.area == pytest.approx(10000.0, rel=1e-9)
