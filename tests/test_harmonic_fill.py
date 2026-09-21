"""Unit tests for `bed.py`'s harmonic (Laplace) gap fill.

The fill supplies 16-49% of the reported submarine volume inside the change
polygons, so its correctness and its sign guarantee are load-bearing. These
tests pin both against known solutions rather than against a prior run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import bed


def _boundary_ring(shape: tuple[int, int]) -> np.ndarray:
    known = np.zeros(shape, dtype=bool)
    known[0, :] = known[-1, :] = known[:, 0] = known[:, -1] = True
    return known


def _laplace_residual(
    values: np.ndarray, known: np.ndarray, domain: np.ndarray
) -> float:
    """Max |cell - mean(in-domain neighbours)| over solved cells."""
    total = np.zeros_like(values)
    count = np.zeros(values.shape, dtype=np.float32)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        shifted = np.roll(values, shift, axis=axis)
        inside = np.roll(domain, shift, axis=axis)
        edge: list[slice | int] = [slice(None), slice(None)]
        edge[axis] = 0 if shift == 1 else -1
        shifted[tuple(edge)] = 0.0
        inside[tuple(edge)] = False
        total += shifted * inside
        count += inside
    solve = domain & ~known & (count > 0)
    target = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return float(np.abs(values[solve] - target[solve]).max()) if solve.any() else 0.0


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("linear", lambda xx, yy: 0.5 * xx - 0.25 * yy),
        ("saddle", lambda xx, yy: (xx**2 - yy**2) / 160.0),
        ("constant", lambda xx, yy: np.full_like(xx, -42.0)),
    ],
)
def test_reproduces_known_harmonic_functions(name, field):
    """A harmonic field is its own solution, so the fill must recover it."""
    n = 96
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    exact = field(xx, yy).astype(np.float32)

    domain = np.ones((n, n), dtype=bool)
    known = _boundary_ring((n, n))
    values = np.where(known, exact, 0.0).astype(np.float32)

    out = bed.harmonic_fill(values, known, domain)

    span = float(exact.max() - exact.min()) or 1.0
    assert np.abs(out - exact)[~known].max() < 0.02 * span, name


def test_known_cells_are_never_modified():
    n = 64
    rng = np.random.default_rng(0)
    domain = np.ones((n, n), dtype=bool)
    known = _boundary_ring((n, n))
    known[rng.random((n, n)) < 0.05] = True
    values = np.where(known, -rng.uniform(0, 500, (n, n)), 0.0).astype(np.float32)

    out = bed.harmonic_fill(values.copy(), known, domain)

    np.testing.assert_allclose(out[known], values[known])


def test_the_solve_is_exact_not_merely_relaxed():
    """A long channel is the hard case for iterative fills.

    The sparse solve reaches the true harmonic solution rather than
    approaching it, so the discrete Laplacian residual should sit near the
    solver tolerance, and the profile along a uniform channel should be the
    exact linear ramp between its two ends.
    """
    height, width = 200, 400
    domain = np.zeros((height, width), dtype=bool)
    domain[95:105, :] = True
    known = np.zeros((height, width), dtype=bool)
    known[95:105, 0] = known[95:105, -1] = True
    values = np.zeros((height, width), dtype=np.float32)
    values[95:105, 0] = -400.0
    values[95:105, -1] = -10.0

    out = bed.harmonic_fill(values.copy(), known, domain)

    assert _laplace_residual(out, known, domain) < 1e-2
    exact_profile = -400.0 + (np.arange(width) / (width - 1)) * 390.0
    assert np.abs(out[100, :] - exact_profile).max() < 0.5


def test_no_free_parameter_changes_the_answer():
    """Tightening the tolerance must not move the solution meaningfully.

    This is the property that distinguishes a convergence setting from a
    physical one, and the reason a block-averaging fill was rejected: its
    block size moved published volumes by several percent.
    """
    shape = (96, 96)
    rng = np.random.default_rng(4)
    domain = np.ones(shape, dtype=bool)
    known = _boundary_ring(shape)
    values = np.zeros(shape, dtype=np.float32)
    values[known] = -np.abs(rng.normal(0, 200, int(known.sum()))).astype(np.float32)

    loose = bed.harmonic_fill(values.copy(), known, domain)
    original = bed.CG_TOLERANCE
    try:
        bed.CG_TOLERANCE = 1e-10
        tight = bed.harmonic_fill(values.copy(), known, domain)
    finally:
        bed.CG_TOLERANCE = original

    assert np.abs(loose - tight)[domain & ~known].max() < 0.01


def test_failure_to_converge_raises_rather_than_returning_garbage():
    shape = (128, 128)
    domain = np.ones(shape, dtype=bool)
    known = _boundary_ring(shape)
    values = np.zeros(shape, dtype=np.float32)
    values[known] = -100.0

    original = bed.CG_MAX_ITERATIONS
    try:
        bed.CG_MAX_ITERATIONS = 1
        with pytest.raises(RuntimeError, match="did not converge"):
            bed.harmonic_fill(values.copy(), known, domain)
    finally:
        bed.CG_MAX_ITERATIONS = original


def test_marine_solve_never_interpolates_above_sea_level():
    """All boundaries <= 0 implies all interior <= 0, for the converged solve.

    This is a property of `harmonic_fill` itself and still holds. What changed
    on 2026-09-16 is that `_make_merged_bed` no longer feeds it an all-<= 0
    boundary set: an inversion value above sea level inside the coastline's
    reach is now kept as a known cell, so the maximum principle no longer
    applies to the marine solve as a whole. See
    `test_a_positive_boundary_lifts_the_interior_above_zero` for the
    consequence, and the `always_sea` clamp in `_make_merged_bed` for how
    permanently-open water is protected instead.
    """
    rng = np.random.default_rng(11)
    for _ in range(25):
        shape = (int(rng.integers(48, 100)), int(rng.integers(48, 100)))
        domain = np.ones(shape, dtype=bool)
        known = rng.random(shape) < 0.03
        known |= _boundary_ring(shape)
        values = np.zeros(shape, dtype=np.float32)
        values[known] = -np.abs(rng.normal(0, 250, int(known.sum()))).astype(np.float32)

        out = bed.harmonic_fill(values, known, domain)

        assert out[domain & ~known].max() <= 0.0


def test_adversarial_gradient_does_not_overshoot_above_zero():
    """One boundary far below sea level, the rest exactly at it."""
    shape = (64, 64)
    domain = np.ones(shape, dtype=bool)
    known = _boundary_ring(shape)
    values = np.zeros(shape, dtype=np.float32)
    values[0, :] = -1000.0

    out = bed.harmonic_fill(values, known, domain)

    assert out[domain & ~known].max() <= 0.0


def test_cells_outside_the_domain_are_preserved():
    """The sparse solve writes only in-domain unknowns, so this now holds.

    Under the previous multigrid implementation it did not: the initial
    guess overwrote every non-`known` cell in or out of domain (review
    finding F18). The rewrite fixes that, and the docstring's promise is
    now true.
    """
    shape = (64, 64)
    domain = np.zeros(shape, dtype=bool)
    domain[10:50, 10:50] = True
    known = np.zeros(shape, dtype=bool)
    known[10, 10:50] = known[49, 10:50] = True
    values = np.full(shape, 999.0, dtype=np.float32)
    values[known] = -100.0

    out = bed.harmonic_fill(values.copy(), known, domain)

    assert np.all(out[~domain] == 999.0)
    assert out[domain & ~known].max() <= 0.0


def test_unconstrained_components_become_nan():
    """A domain island with no known cell has no boundary condition at all."""
    shape = (64, 64)
    domain = np.zeros(shape, dtype=bool)
    domain[5:20, 5:20] = True  # constrained
    domain[40:55, 40:55] = True  # island, no known cell
    known = np.zeros(shape, dtype=bool)
    known[5, 5:20] = True
    values = np.zeros(shape, dtype=np.float32)
    values[known] = -50.0

    filled = bed.harmonic_fill(values.copy(), known, domain)
    out = bed._drop_unconstrained_components(filled, known, domain)

    assert np.isnan(out[45, 45])
    assert not np.isnan(out[10, 10])


def test_a_positive_boundary_lifts_the_interior_above_zero():
    """One positive known cell is enough to break the <= 0 guarantee.

    This is why `_make_merged_bed` clamps interpolated cells in the
    always-sea zone rather than relying on the maximum principle: admitting
    a legitimate above-sea-level inversion value at a fjord head propagates
    outward through the harmonic solve. Measured on the real grid
    (2026-09-16) it reached +141 m in permanently-open water before the
    clamp was added.
    """
    shape = (48, 48)
    domain = np.ones(shape, dtype=bool)
    known = _boundary_ring(shape)
    values = np.zeros(shape, dtype=np.float32)
    values[known] = -100.0
    values[0, 16:32] = 60.0  # a fjord head's worth, not a single cell

    out = bed.harmonic_fill(values, known, domain)

    assert out[domain & ~known].max() > 0.0
