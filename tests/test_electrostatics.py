"""Conductor equilibrium, edge effects and capacitance.

These check the solver against results that are known in closed form, and in
particular against ones that exist *only* because of edge effects -- the disc
capacitance and its charge density -- so a solver that quietly assumed uniform
density could not pass.
"""

import math

import numpy as np
import pytest

from emsim.core import Disk, Line, PointCharge, Scene, Sheet, Sphere
from emsim.core.bodies import Material
from emsim.core.electrostatics import build_system
from emsim.core.kernels import coulomb_sum
from emsim.units import EPS0, K_COULOMB


# -- capacitance against closed form ---------------------------------------
def test_isolated_sphere_capacitance_approaches_four_pi_eps0_r():
    R = 0.3
    exact = 4 * math.pi * EPS0 * R
    errs = [abs(1 / build_system([Sphere(radius=R, n_sites=n)]).p_cond[0, 0]
                / exact - 1) for n in (150, 600, 1500)]
    assert errs[-1] < 0.005
    assert errs[0] > errs[1] > errs[2]          # converging, not just close


def test_isolated_disc_capacitance_approaches_eight_eps0_r():
    """C = 8 eps0 R holds only because charge piles up at the rim."""
    R = 0.3
    exact = 8 * EPS0 * R
    errs = [abs(1 / build_system([Disk(radius=R, n_sites=n)]).p_cond[0, 0]
                / exact - 1) for n in (200, 800, 2000)]
    assert errs[-1] < 0.005
    assert errs[0] > errs[1] > errs[2]


# -- the defining properties of a conductor --------------------------------
@pytest.fixture
def charged_sphere():
    s = Scene()
    b = s.add_body(Sphere(radius=0.3, n_sites=900))
    b.charge = 1e-9
    return s, b


def test_field_vanishes_inside_a_conductor(charged_sphere):
    s, b = charged_sphere
    rng = np.random.default_rng(0)
    d = rng.normal(size=(200, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    inner = b.position + d * (0.8 * b.radius * rng.uniform(0, 1, 200)[:, None])
    inside = np.linalg.norm(s.evaluate(inner)["E"], axis=1).mean()
    outside = np.linalg.norm(s.evaluate([[1.0, 0.0, 0.0]])["E"][0])
    assert inside / outside < 0.01


def test_conductor_surface_is_an_equipotential(charged_sphere):
    s, b = charged_sphere
    pos, _ = s.state().body_sites[b.uid]
    V = s.evaluate(pos)["V"]
    assert V.std() / abs(V.mean()) < 0.01


def test_charge_on_a_conducting_sphere_is_uniform(charged_sphere):
    s, b = charged_sphere
    _, q = s.state().body_sites[b.uid]
    assert q.std() / q.mean() < 0.01
    assert q.sum() == pytest.approx(b.charge)


def test_conductor_charge_is_conserved_exactly():
    s = Scene()
    a = s.add_body(Sphere(radius=0.25, position=[-0.5, 0, 0], n_sites=200))
    c = s.add_body(Disk(radius=0.3, position=[0.6, 0, 0], n_sites=200))
    a.charge, c.charge = 3e-9, -7e-9
    s.add_charge(PointCharge(q=2e-9, position=[0.0, 0.4, 0.0]))
    st = s.state()
    assert st.body_sites[a.uid][1].sum() == pytest.approx(3e-9)
    assert st.body_sites[c.uid][1].sum() == pytest.approx(-7e-9)


def test_an_uncharged_conductor_still_develops_induced_charge():
    """Net zero, but separated: the near face goes negative, the far positive."""
    s = Scene()
    plate = s.add_body(Sheet(width=0.8, height=0.8, position=[0, 0, 0],
                             axis=[0, 0, 1], n_sites=400))
    s.add_charge(PointCharge(q=5e-9, position=[0.0, 0.0, 0.35]))
    pos, q = s.state().body_sites[plate.uid]
    assert q.sum() == pytest.approx(0.0, abs=1e-20)
    r = np.linalg.norm(pos[:, :2], axis=1)
    assert q[r < 0.15].mean() < 0            # under the charge: opposite sign
    assert q[r > 0.5].mean() > 0             # far corners: same sign


# -- edge effects ----------------------------------------------------------
def test_disc_charge_density_follows_the_inverse_square_root_edge_law():
    R = 0.3
    b = Disk(radius=R, n_sites=1600)
    sysm = build_system([b])
    q, _ = sysm.solve(np.array([1e-9]))
    r = np.linalg.norm(sysm.pos[:, :2], axis=1)
    inner = r < 0.95 * R
    ratio = q[inner] * np.sqrt(R**2 - r[inner] ** 2)     # should be constant
    assert ratio.std() / ratio.mean() < 0.06
    # and the pile-up is large, not a rounding effect
    assert q[r > 0.9 * R].mean() / q[r < 0.2 * R].mean() > 2.0


def test_parallel_plate_capacitance_exceeds_the_textbook_value_by_fringing():
    W, A = 0.6, 0.36

    def cap(gap):
        p1 = Sheet(width=W, height=W, position=[0, 0, -gap / 2], n_sites=700)
        p2 = Sheet(width=W, height=W, position=[0, 0, gap / 2], n_sites=700)
        return build_system([p1, p2]).capacitance(p1.uid, p2.uid)

    ratios = [cap(g) / (EPS0 * A / g) for g in (0.30, 0.15, 0.06)]
    assert all(r > 1.0 for r in ratios)          # fringing always adds
    assert ratios[0] > ratios[1] > ratios[2]     # and matters less as d shrinks
    assert ratios[-1] < 1.5                      # but is a correction, not a rewrite


# -- capacitance is a property of the geometry -----------------------------
def test_capacitance_is_symmetric_and_independent_of_the_charge_present():
    s = Scene()
    a = s.add_body(Disk(radius=0.3, position=[0, 0, -0.1], n_sites=300))
    b = s.add_body(Disk(radius=0.3, position=[0, 0, 0.1], n_sites=300))
    a.charge, b.charge = 4e-9, -4e-9
    c1 = s.measure(a.uid, b.uid)["C"]
    assert s.measure(b.uid, a.uid)["C"] == pytest.approx(c1)
    a.charge, b.charge = -1e-8, 3e-9
    s.notify()
    assert s.measure(a.uid, b.uid)["C"] == pytest.approx(c1)


def test_measured_capacitance_matches_the_geometric_one_for_equal_and_opposite():
    s = Scene()
    a = s.add_body(Sheet(width=0.5, height=0.5, position=[0, 0, -0.08],
                         n_sites=400))
    b = s.add_body(Sheet(width=0.5, height=0.5, position=[0, 0, 0.08],
                         n_sites=400))
    a.charge, b.charge = 2e-9, -2e-9
    m = s.measure(a.uid, b.uid)
    assert m["balanced"]
    assert m["C_measured"] == pytest.approx(m["C"], rel=1e-9)
    assert m["dV"] > 0                       # positive plate is at higher V
    assert m["energy"] == pytest.approx(0.5 * m["C"] * m["dV"] ** 2)


def test_bringing_plates_closer_raises_the_capacitance():
    def cap(gap):
        s = Scene()
        a = s.add_body(Disk(radius=0.3, position=[0, 0, -gap / 2], n_sites=250))
        b = s.add_body(Disk(radius=0.3, position=[0, 0, gap / 2], n_sites=250))
        a.charge, b.charge = 1e-9, -1e-9
        return s.measure(a.uid, b.uid)

    near, far = cap(0.05), cap(0.4)
    assert near["C"] > far["C"]
    assert abs(near["dV"]) < abs(far["dV"])   # same charge, closer => less volts


def test_measure_rejects_insulators_and_self_pairs():
    s = Scene()
    a = s.add_body(Sphere(radius=0.2, position=[-0.4, 0, 0]))
    ins = s.add_body(Sphere(radius=0.2, position=[0.4, 0, 0],
                            material=Material.INSULATOR))
    assert s.measure(a.uid, ins.uid) is None
    assert s.measure(a.uid, a.uid) is None


# -- insulators ------------------------------------------------------------
def test_insulator_charge_stays_uniform_and_in_place():
    s = Scene()
    b = s.add_body(Sphere(radius=0.3, material=Material.INSULATOR, n_sites=500))
    b.charge = 6e-9
    s.add_charge(PointCharge(q=9e-9, position=[0.8, 0, 0]))   # would polarise
    pos, q = s.state().body_sites[b.uid]
    assert q.std() == pytest.approx(0.0, abs=1e-24)
    assert q.sum() == pytest.approx(6e-9)


def test_uniformly_charged_insulating_ball_reproduces_gauss_law():
    R, Q = 0.3, 5e-9
    s = Scene()
    b = s.add_body(Sphere(radius=R, material=Material.INSULATOR, n_sites=4000))
    b.charge = Q
    # outside: indistinguishable from a point charge
    out = s.evaluate([[1.2, 0.0, 0.0]])
    assert out["E"][0][0] == pytest.approx(K_COULOMB * Q / 1.2**2, rel=0.01)
    assert out["V"][0] == pytest.approx(K_COULOMB * Q / 1.2, rel=0.01)
    # inside: E grows linearly with r
    r = 0.15
    inn = s.evaluate([[r, 0.0, 0.0]])
    assert inn["E"][0][0] == pytest.approx(K_COULOMB * Q * r / R**3, rel=0.10)


# -- relaxation ------------------------------------------------------------
def test_relaxation_starts_uniform_inside_and_ends_at_equilibrium():
    s = Scene()
    b = s.add_body(Sphere(radius=0.3, n_sites=300))
    b.charge = 1e-9

    b.relax_t = 1.0
    s.invalidate()
    pos_end, q_end = s.state().body_sites[b.uid]
    assert np.linalg.norm(pos_end - b.position, axis=1) == pytest.approx(
        np.full(len(pos_end), 0.3))

    b.relax_t = 0.0
    s.invalidate()
    pos0, q0 = s.state().body_sites[b.uid]
    assert q0.std() == pytest.approx(0.0, abs=1e-24)       # uniformly spread
    assert q0.sum() == pytest.approx(b.charge)
    r0 = np.linalg.norm(pos0 - b.position, axis=1)
    assert r0.max() < 0.3 and r0.min() < 0.05              # filling the volume
    assert q0.sum() == pytest.approx(q_end.sum())          # charge conserved

    b.relax_t = 0.5
    s.invalidate()
    r_mid = np.linalg.norm(s.state().body_sites[b.uid][0] - b.position, axis=1)
    assert r0.mean() < r_mid.mean() < 0.3                  # on its way out


def test_depositing_charge_on_a_conductor_restarts_the_relaxation():
    s = Scene()
    b = s.add_body(Sphere(radius=0.3))
    b.relax_t = 1.0
    s.deposit_charge(b.uid, 2e-9)
    assert b.charge == pytest.approx(2e-9)
    assert b.relax_t == 0.0


def test_depositing_on_an_insulator_does_not_animate():
    s = Scene()
    b = s.add_body(Sphere(radius=0.3, material=Material.INSULATOR))
    s.deposit_charge(b.uid, 2e-9)
    assert b.relax_t == 1.0


# -- forces ----------------------------------------------------------------
def test_body_force_excludes_its_own_charge():
    s = Scene()
    b = s.add_body(Sphere(radius=0.25, n_sites=200))
    b.charge = 4e-9
    assert np.linalg.norm(s.body_forces()[b.uid]) < 1e-18   # alone => no force


def test_force_on_a_charged_sphere_matches_the_image_charge_solution():
    """Exact classical result, and one a uniform-charge model cannot reproduce.

        F = k [ qQ/d^2  -  q^2 R^3 (2d^2 - R^2) / (d^2 (d^2 - R^2)^2) ]

    The second term is the pull of the charge the point charge induces on the
    sphere; it cancels about 7% of the bare repulsion at this spacing.
    """
    q, Q, R, d = 8e-9, 4e-9, 0.25, 1.0
    s = Scene()
    b = s.add_body(Sphere(radius=R, n_sites=800))
    b.charge = Q
    s.add_charge(PointCharge(q=q, position=[d, 0.0, 0.0]))

    F = s.body_forces()[b.uid]
    exact = K_COULOMB * (q * Q / d**2
                         - q * q * R**3 * (2 * d * d - R * R)
                         / (d * d * (d * d - R * R) ** 2))
    assert F[0] < 0                      # like charges: pushed away from +x
    assert np.linalg.norm(F) == pytest.approx(exact, rel=0.005)
    # and it is meaningfully below the polarisation-free answer
    assert np.linalg.norm(F) < 0.96 * K_COULOMB * q * Q / d**2


def test_point_charge_is_attracted_to_its_image_in_a_grounded_plate():
    """A classic: the induced charge pulls the point charge toward the plate."""
    s = Scene()
    s.add_body(Sheet(width=3.0, height=3.0, position=[0, 0, 0], axis=[0, 0, 1],
                     n_sites=900))
    c = s.add_charge(PointCharge(q=1e-9, position=[0.0, 0.0, 0.25]))
    F = s.forces()[0]
    assert F[2] < 0                                   # pulled toward the plate
    image = K_COULOMB * 1e-9 * 1e-9 / (2 * 0.25) ** 2
    assert abs(F[2]) == pytest.approx(image, rel=0.15)


# -- containment (used to keep field glyphs out of solid geometry) ---------
def test_contains_marks_the_interior_of_each_body():
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [2.0, 0.0, 0.0]])

    assert Sphere(radius=0.3).contains(pts).tolist() == [True, False, False]
    # A flat body is a closed set of zero thickness: a point exactly on it
    # counts, one just off it does not until `pad` gives the plate a slab.
    plate = Sheet(width=1.0, height=1.0)
    assert plate.contains(np.array([[0.0, 0.0, 0.0]]))[0]
    assert not plate.contains(np.array([[0.0, 0.0, 0.001]]))[0]
    assert plate.contains(np.array([[0.0, 0.0, 0.001]]), pad=0.01)[0]
    assert Sheet(width=1.0, height=1.0).contains(pts, pad=0.05).tolist() == \
        [True, False, False]
    assert Disk(radius=0.4).contains(pts, pad=0.05).tolist() == \
        [True, False, False]
    assert Line(length=1.0, wire_radius=0.02).contains(pts, pad=0.01).tolist() \
        == [True, True, False]


# -- the caching that keeps interaction fast -------------------------------
def test_moving_a_point_charge_does_not_refactorise_the_conductors():
    """The O(N^3) inverse must survive a drag that leaves geometry alone."""
    s = Scene()
    s.add_body(Sheet(width=0.8, height=0.8, n_sites=120))
    c = s.add_charge(PointCharge(q=3e-9, position=[0.0, 0.0, 0.4]))
    first = s.state().system

    c.position[2] = 0.25
    s.invalidate()
    assert s.state().system is first          # same object: reused, not rebuilt

    # ...but the answer still changes, so it is genuinely re-solved
    s.invalidate()
    near = s.state().body_sites[list(s.state().body_sites)[0]][1]
    c.position[2] = 1.5
    s.invalidate()
    far = s.state().body_sites[list(s.state().body_sites)[0]][1]
    assert np.abs(near).max() > 3 * np.abs(far).max()


def test_moving_a_conductor_does_refactorise():
    s = Scene()
    b = s.add_body(Sheet(width=0.8, height=0.8, n_sites=120))
    first = s.state().system
    b.position = np.array([0.0, 0.0, 0.3])
    s.invalidate()
    assert s.state().system is not first


def test_coarsening_during_a_drag_keeps_capacitance_in_the_right_ballpark():
    s = Scene()
    a = s.add_body(Disk(radius=0.3, position=[0, 0, -0.1], n_sites=400))
    b = s.add_body(Disk(radius=0.3, position=[0, 0, 0.1], n_sites=400))
    full = s.measure(a.uid, b.uid)["C"]
    s.site_scale = 0.45
    s.invalidate()
    coarse = s.measure(a.uid, b.uid)["C"]
    assert coarse == pytest.approx(full, rel=0.05)


# -- depositing charge -----------------------------------------------------
def test_deposits_accumulate_and_land_on_the_body():
    s = Scene()
    b = s.add_body(Sphere(radius=0.3, n_sites=200))
    s.deposit_charge(b.uid, 5e-9)
    s.deposit_charge(b.uid, 5e-9)
    s.deposit_charge(b.uid, -2e-9)
    assert b.charge == pytest.approx(8e-9)
    b.relax_t = 1.0
    s.invalidate()
    assert s.state().body_sites[b.uid][1].sum() == pytest.approx(8e-9)


def test_switching_material_moves_the_charge_between_surface_and_volume():
    s = Scene()
    b = s.add_body(Sphere(radius=0.3, n_sites=300))
    b.charge = 4e-9

    r = np.linalg.norm(s.state().body_sites[b.uid][0] - b.position, axis=1)
    assert r == pytest.approx(np.full(len(r), 0.3))        # conductor: surface

    b.material = Material.INSULATOR
    s.invalidate()
    r = np.linalg.norm(s.state().body_sites[b.uid][0] - b.position, axis=1)
    assert r.min() < 0.1 and r.max() < 0.3                 # insulator: volume
    assert s.state().body_sites[b.uid][1].std() == pytest.approx(0.0, abs=1e-24)


def test_a_scene_of_only_bodies_still_evaluates_fields():
    """Regression: the field layers used to bail when no point charge existed."""
    s = Scene()
    b = s.add_body(Sphere(radius=0.3, n_sites=200))
    b.charge = 5e-9
    out = s.evaluate([[1.0, 0.0, 0.0]])
    assert np.linalg.norm(out["E"][0]) > 0
    assert out["V"][0] > 0


# -- the unified object list -----------------------------------------------
def test_objects_lists_charges_and_bodies_together_in_creation_order():
    """The panel shows one list, so the model offers one."""
    s = Scene()
    a = s.add_body(Sphere(radius=0.2))
    b = s.add_charge(PointCharge(q=1e-9, position=[0, 0, 0]))
    c = s.add_body(Disk(radius=0.3))
    d = s.add_charge(PointCharge(q=-1e-9, position=[1, 0, 0]))
    assert [o.uid for o in s.objects()] == [a.uid, b.uid, c.uid, d.uid]
    assert len(s.objects()) == len(s.charges) + len(s.bodies)


def test_objects_order_survives_removal():
    s = Scene()
    a = s.add_body(Sphere(radius=0.2))
    b = s.add_charge(PointCharge(q=1e-9, position=[0, 0, 0]))
    c = s.add_body(Disk(radius=0.3))
    s.remove_charge(b.uid)
    assert [o.uid for o in s.objects()] == [a.uid, c.uid]
    s.remove_body(a.uid)
    assert [o.uid for o in s.objects()] == [c.uid]


def test_objects_is_empty_for_an_empty_scene():
    assert Scene().objects() == []


def test_every_object_exposes_what_the_list_shows():
    """One table, so both kinds need position, a charge and a label."""
    s = Scene()
    s.add_charge(PointCharge(q=2e-9, position=[0.1, 0, 0]))
    s.add_body(Disk(radius=0.3))
    for o in s.objects():
        assert isinstance(o.label, str) and o.label
        assert o.position.shape == (3,)
        charge = o.charge if hasattr(o, "material") else o.q
        assert isinstance(float(charge), float)
        assert s.by_uid(o.uid) is o
