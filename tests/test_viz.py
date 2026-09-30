"""Geometry, colormaps and normalisations (no GL context required)."""

import numpy as np
import pytest

from emsim.viz.colormaps import DIVERGING, SEQUENTIAL, lut, map_rgb
from emsim.viz.geometry import arrow_soup, box_lines, rotations_to, unit_arrow
from emsim.viz.norms import LinearNorm, LogNorm, SymLogNorm, robust_range


# -- geometry --------------------------------------------------------------
@pytest.mark.parametrize("d", [
    [0, 0, 1], [0, 0, -1], [1, 0, 0], [0, -1, 0], [0.3, -0.5, 0.8], [-1, -1, -1],
])
def test_rotation_maps_z_onto_the_direction(d):
    d = np.array(d, dtype=float)
    d /= np.linalg.norm(d)
    R = rotations_to(d[None])[0]
    assert R @ np.array([0.0, 0.0, 1.0]) == pytest.approx(d)
    assert R @ R.T == pytest.approx(np.eye(3))       # orthonormal
    assert np.linalg.det(R) == pytest.approx(1.0)    # proper rotation, no flip


def test_zero_direction_does_not_produce_nans():
    R = rotations_to(np.zeros((1, 3)))
    assert np.isfinite(R).all()


def test_unit_arrow_is_a_closed_shell():
    n = 8
    v, f = unit_arrow(n)
    assert v.shape == (3 * n + 2, 3)
    assert f.shape == (6 * n, 3)
    assert v[:, 2].min() == pytest.approx(0.0)
    assert v[:, 2].max() == pytest.approx(1.0)

    # Watertight: every directed edge appears once, so every undirected edge
    # is shared by exactly two oppositely-wound faces.
    directed = [(int(a), int(b)) for tri in f
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))]
    assert len(directed) == len(set(directed))
    assert {(b, a) for a, b in directed} == set(directed)


def test_unit_arrow_is_wound_outward():
    """Divergence theorem: consistent outward winding gives the true volume."""
    from emsim.viz.geometry import _HEAD_FRAC, _HEAD_R, _SHAFT_R

    n = 24
    v, f = unit_arrow(n)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    volume = np.einsum("ij,ij->i", np.cross(a, b), c).sum() / 6.0

    ngon = lambda r: 0.5 * n * r**2 * np.sin(2 * np.pi / n)   # noqa: E731
    expected = (ngon(_SHAFT_R) * (1.0 - _HEAD_FRAC)
                + ngon(_HEAD_R) * _HEAD_FRAC / 3.0)
    assert volume == pytest.approx(expected, rel=1e-6)
    assert volume > 0        # negative would mean every face points inward


def test_unit_arrow_has_no_degenerate_faces():
    v, f = unit_arrow(10)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert area.min() > 1e-9


def test_arrow_soup_places_tips_at_origin_plus_length_times_direction():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]])
    dirs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    lengths = np.array([2.0, 0.5])
    tris, cols = arrow_soup(origins, dirs, lengths, np.full(2, 0.2),
                            np.tile([1.0, 0, 0, 1.0], (2, 1)))
    assert tris.shape[1:] == (3, 3) and cols.shape[1:] == (3, 4)
    per = len(tris) // 2
    for i, (o, d, L) in enumerate(zip(origins, dirs, lengths)):
        pts = tris[i * per:(i + 1) * per].reshape(-1, 3)
        assert pts.max(axis=0) == pytest.approx(np.maximum(o, o + d * L), abs=0.11)
        # the extreme point along d is the tip
        assert (pts @ d).max() == pytest.approx((o + d * L) @ d)


def test_arrow_soup_handles_an_empty_field():
    tris, cols = arrow_soup(np.zeros((0, 3)), np.zeros((0, 3)),
                            np.zeros(0), np.zeros(0), np.zeros((0, 4)))
    assert tris.shape == (0, 3, 3) and cols.shape == (0, 3, 4)


def test_shading_stays_within_the_ambient_floor():
    """Baked shading must not swing colours far from their colormap value."""
    n = 40
    rng = np.random.default_rng(0)
    base = np.tile([0.8, 0.4, 0.2, 1.0], (n, 1))
    _, cols = arrow_soup(rng.normal(size=(n, 3)), rng.normal(size=(n, 3)),
                         np.full(n, 0.2), np.full(n, 0.2), base, ambient=0.72)
    ratio = cols[:, :, 0] / 0.8
    assert ratio.min() >= 0.72 - 1e-6
    assert ratio.max() <= 1.0 + 1e-6
    assert cols[:, :, 3] == pytest.approx(1.0)      # alpha untouched


def test_box_lines_span_the_domain():
    v = box_lines(2.5)
    assert v.shape == (24, 3)
    assert set(np.unique(v)) == {-2.5, 2.5}


# -- colormaps -------------------------------------------------------------
@pytest.mark.parametrize("name", SEQUENTIAL + DIVERGING)
def test_lut_is_a_full_uint8_table(name):
    t = lut(name)
    assert t.shape == (256, 3) and t.dtype == np.uint8


def test_diverging_maps_are_light_in_the_middle_and_saturated_at_the_ends():
    for name in DIVERGING:
        lo, mid, hi = map_rgb(name, np.array([0.0, 0.5, 1.0]))
        assert mid.mean() > lo.mean() and mid.mean() > hi.mean()
        assert lo[2] > lo[0]        # negative end is blue
        assert hi[0] > hi[2]        # positive end is red


def test_map_rgb_clips_out_of_range_input():
    assert (map_rgb("viridis", np.array([-5.0])) == lut("viridis")[0]).all()
    assert (map_rgb("viridis", np.array([5.0])) == lut("viridis")[-1]).all()
    assert np.isfinite(map_rgb("viridis", np.array([np.nan]))).all()


# -- norms -----------------------------------------------------------------
def test_linear_norm_endpoints_and_clipping():
    n = LinearNorm(-3.0, 7.0)
    assert n(-3.0) == pytest.approx(0.0)
    assert n(7.0) == pytest.approx(1.0)
    assert n(2.0) == pytest.approx(0.5)
    assert n(-99.0) == 0.0 and n(99.0) == 1.0


def test_linear_norm_survives_a_degenerate_range():
    assert LinearNorm(5.0, 5.0)(5.0) == 0.0


def test_log_norm_is_linear_in_the_exponent():
    n = LogNorm(1e-2, 1e3)
    assert n(1e-2) == pytest.approx(0.0)
    assert n(1e3) == pytest.approx(1.0)
    assert n(10.0) == pytest.approx(0.6)
    assert [v for _, v in n.ticks()] == pytest.approx([1e-2, 1e-1, 1, 10, 100, 1e3])


def test_log_norm_ticks_include_the_ends_when_there_are_few_decades():
    fracs = [f for f, _ in LogNorm(3.0, 40.0).ticks()]
    assert min(fracs) == pytest.approx(0.0) and max(fracs) == pytest.approx(1.0)


def test_symlog_is_centred_on_zero_and_antisymmetric():
    n = SymLogNorm(1e4, 1.0)
    assert n(0.0) == pytest.approx(0.5)
    assert n(1e4) == pytest.approx(1.0)
    assert n(-1e4) == pytest.approx(0.0)
    for v in (0.3, 5.0, 250.0, 9999.0):
        assert n(v) + n(-v) == pytest.approx(1.0)
        assert n(v) > 0.5 > n(-v)


def test_symlog_is_monotonic_across_the_linear_seam():
    n = SymLogNorm(1e3, 2.0)
    xs = np.concatenate([-np.logspace(3, -3, 200), [0.0], np.logspace(-3, 3, 200)])
    ys = n(xs)
    assert np.all(np.diff(ys) >= -1e-12)


def test_norm_ticks_are_inside_the_bar():
    for n in (LinearNorm(-2, 9), LogNorm(1e-3, 1e2), SymLogNorm(500.0, 0.5)):
        for frac, _ in n.ticks():
            assert 0.0 <= frac <= 1.0


def test_robust_range_ignores_outliers_and_nans():
    v = np.concatenate([np.linspace(1.0, 2.0, 1000), [1e9], [np.nan]])
    lo, hi = robust_range(v, 2.0, 98.0)
    assert 1.0 <= lo < hi < 3.0


def test_robust_range_on_empty_input():
    assert robust_range(np.array([])) == (0.0, 1.0)


# -- force arrow scale ------------------------------------------------------
def test_force_arrow_length_is_proportional_to_the_magnitude():
    from emsim.viz.layers import (FORCE_REFERENCE, FORCE_REF_LENGTH,
                                  force_arrow_length)
    d = 1.0
    assert force_arrow_length(FORCE_REFERENCE, d) == pytest.approx(
        FORCE_REF_LENGTH * d)
    # doubling the force doubles the arrow; quartering it quarters the arrow
    base = force_arrow_length(FORCE_REFERENCE, d)
    assert force_arrow_length(2 * FORCE_REFERENCE, d) == pytest.approx(2 * base)
    assert force_arrow_length(FORCE_REFERENCE / 4, d) == pytest.approx(base / 4)
    assert force_arrow_length(0.0, d) == pytest.approx(0.0)


def test_force_arrow_scale_does_not_depend_on_the_rest_of_the_scene():
    """The whole point of the fixed scale: no renormalisation to the max."""
    from emsim.core import PointCharge, Scene
    from emsim.viz.layers import force_arrow_length

    def length_on_first(extra=None):
        s = Scene()
        s.add_charge(PointCharge(q=10e-9, position=[-0.35, 0, 0]))
        s.add_charge(PointCharge(q=-10e-9, position=[0.35, 0, 0]))
        if extra is not None:
            s.add_charge(extra)
        mag = np.linalg.norm(s.forces(), axis=1)
        return force_arrow_length(mag, 1.0)[0], mag[0]

    alone, _ = length_on_first()
    # a huge charge far away barely changes the force on charge 0, so its
    # arrow must barely change -- under the old normalisation it would have
    # collapsed to almost nothing
    with_giant, _ = length_on_first(
        PointCharge(q=5e-6, position=[0.0, 0.0, 60.0]))
    assert with_giant == pytest.approx(alone, rel=2e-3)


def test_moving_charges_closer_grows_the_arrow_as_inverse_square():
    from emsim.core import PointCharge, Scene
    from emsim.viz.layers import force_arrow_length

    def arrow_at(sep):
        s = Scene()
        s.add_charge(PointCharge(q=10e-9, position=[-sep / 2, 0, 0]))
        s.add_charge(PointCharge(q=-10e-9, position=[sep / 2, 0, 0]))
        return force_arrow_length(np.linalg.norm(s.forces(), axis=1), 1.0)[0]

    assert arrow_at(0.4) / arrow_at(0.8) == pytest.approx(4.0, rel=1e-9)
    assert arrow_at(1.2) / arrow_at(0.6) == pytest.approx(0.25, rel=1e-9)


def test_force_arrow_length_is_clamped_for_near_coincident_charges():
    from emsim.viz.layers import FORCE_MAX_LENGTH, force_arrow_length
    huge = force_arrow_length(1e30, 2.0)
    assert huge == pytest.approx(FORCE_MAX_LENGTH * 2.0)


def test_fit_gain_puts_the_largest_force_at_the_reference_length():
    from emsim.viz.layers import (FORCE_REF_LENGTH, force_arrow_length,
                                  force_gain_for)
    mags = np.array([3e-4, 1.1e-3, 7e-5])
    g = force_gain_for(mags)
    assert force_arrow_length(mags.max(), 1.0, g) == pytest.approx(
        FORCE_REF_LENGTH)
    # and it leaves the ratios between arrows untouched
    lengths = force_arrow_length(mags, 1.0, g)
    assert lengths / lengths[0] == pytest.approx(mags / mags[0])


def test_fit_gain_handles_a_scene_with_no_force():
    from emsim.viz.layers import force_gain_for
    assert force_gain_for(np.array([])) == 1.0
    assert force_gain_for(np.zeros(3)) == 1.0


# -- where a force arrow is anchored ---------------------------------------
def _plates(qa, qb, gap=0.36, radius=0.45):
    from emsim.core import Disk, Scene
    s = Scene()
    a = s.add_body(Disk(radius=radius, position=[0, 0, -gap / 2],
                        axis=[0, 0, 1], n_sites=300))
    b = s.add_body(Disk(radius=radius, position=[0, 0, gap / 2],
                        axis=[0, 0, 1], n_sites=300))
    a.charge, b.charge = qa, qb
    return s, a, b


def _arrow(scene, body):
    """The segment actually drawn for the net force on ``body``."""
    from emsim.viz.layers import force_anchor_offset, force_arrow_length
    F = scene.body_forces()[body.uid]
    mag = np.linalg.norm(F)
    d = F / mag
    start = body.position + d * force_anchor_offset(body, d, scene)
    return start, start + d * force_arrow_length(mag, 1.0, 1.0)


def test_force_anchor_is_measured_along_the_force_not_across_the_body():
    """A disc's `radius` is its size *across* the plate.

    Using it as an along-the-force offset pushed a capacitor plate's arrow a
    whole plate radius away -- past the other plate -- which made attraction
    render exactly like repulsion.
    """
    from emsim.core import Scene, Sphere
    from emsim.viz.layers import force_anchor_offset

    s, a, _ = _plates(4e-9, -4e-9)
    normal, in_plane = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    assert force_anchor_offset(a, normal, s) == pytest.approx(0.0, abs=1e-9)
    # A few percent short of the true radius: the extent is measured from the
    # discrete sites, and the outermost ones do not land exactly on +x.
    assert force_anchor_offset(a, in_plane, s) == pytest.approx(0.45, rel=0.05)

    s2 = Scene()
    ball = s2.add_body(Sphere(radius=0.3, n_sites=200))
    ball.charge = 5e-9
    assert force_anchor_offset(ball, normal, s2) == pytest.approx(0.3, rel=0.02)


def test_attracting_plates_draw_arrows_pointing_at_each_other():
    s, a, b = _plates(4e-9, -4e-9)
    for body, other in ((a, b), (b, a)):
        start, tip = _arrow(s, body)
        near = np.linalg.norm(tip - other.position)
        far = np.linalg.norm(start - other.position)
        assert near < far                      # the arrow closes the gap
        # and it starts on the plate, not somewhere past the other one
        assert start[2] == pytest.approx(body.position[2], abs=1e-9)
        assert abs(tip[2]) < abs(body.position[2]) + 1e-9


def test_repelling_plates_draw_arrows_pointing_apart():
    s, a, b = _plates(4e-9, 4e-9)
    for body, other in ((a, b), (b, a)):
        start, tip = _arrow(s, body)
        assert np.linalg.norm(tip - other.position) > \
            np.linalg.norm(start - other.position)
        assert abs(tip[2]) > abs(body.position[2])


def test_inverting_one_plate_reverses_both_drawn_arrows():
    """The symptom that gave the bug away: flipping a sign changed nothing."""
    s_att, a_att, b_att = _plates(4e-9, -4e-9)
    s_rep, a_rep, b_rep = _plates(4e-9, 4e-9)
    for body_att, body_rep in ((a_att, a_rep), (b_att, b_rep)):
        start, tip = _arrow(s_att, body_att)
        start2, tip2 = _arrow(s_rep, body_rep)
        assert np.sign((tip - start)[2]) == -np.sign((tip2 - start2)[2])


def test_point_charge_arrows_still_start_at_the_ball_surface():
    from emsim.core import PointCharge, Scene
    from emsim.core.entities import radius_for_charge
    from emsim.viz.layers import force_anchor_offset

    s = Scene()
    c = s.add_charge(PointCharge(q=10e-9, position=[-0.35, 0, 0],
                                 radius=radius_for_charge(10e-9)))
    s.add_charge(PointCharge(q=-10e-9, position=[0.35, 0, 0]))
    d = np.array([1.0, 0.0, 0.0])
    assert force_anchor_offset(c, d, s) == pytest.approx(c.radius)
