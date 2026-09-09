"""Unit-system conversions and formatting."""

import pytest

from emsim.units import (GAUSSIAN, SI, UNIT_SYSTEMS, K_COULOMB, Quantity,
                         format_value)


def test_si_is_the_identity():
    for q in Quantity:
        assert SI.to_display(3.0, q) == 3.0


@pytest.mark.parametrize("value_si, q, expected, unit", [
    (1.0, Quantity.LENGTH, 100.0, "cm"),
    (1.0, Quantity.CHARGE, 2.99792458e9, "statC"),
    (1.0, Quantity.FORCE, 1e5, "dyn"),
    (1.0, Quantity.ENERGY, 1e7, "erg"),
    (1.0, Quantity.BFIELD, 1e4, "G"),
    (2.99792458e4, Quantity.EFIELD, 1.0, "statV/cm"),
    (2.99792458e2, Quantity.POTENTIAL, 1.0, "statV"),
])
def test_gaussian_conversions(value_si, q, expected, unit):
    assert GAUSSIAN.to_display(value_si, q) == pytest.approx(expected)
    assert GAUSSIAN.unit_symbol(q) == unit


def test_gaussian_is_dimensionally_self_consistent():
    """F = qE and V = E*d must hold after conversion, not just in SI."""
    q_si, e_si, d_si = 3e-9, 220.0, 0.4
    g = GAUSSIAN
    assert (g.to_display(q_si * e_si, Quantity.FORCE)
            == pytest.approx(g.to_display(q_si, Quantity.CHARGE)
                             * g.to_display(e_si, Quantity.EFIELD)))
    assert (g.to_display(e_si * d_si, Quantity.POTENTIAL)
            == pytest.approx(g.to_display(e_si, Quantity.EFIELD)
                             * g.to_display(d_si, Quantity.LENGTH)))


def test_coulombs_law_in_gaussian_units_has_k_equal_one():
    """1 statC at 1 cm must give exactly 1 statV and 1 statV/cm."""
    q_si = 1.0 / 2.99792458e9        # one statcoulomb
    r_si = 0.01                      # one centimetre
    v_si = K_COULOMB * q_si / r_si
    e_si = K_COULOMB * q_si / r_si**2
    assert GAUSSIAN.to_display(v_si, Quantity.POTENTIAL) == pytest.approx(1.0)
    assert GAUSSIAN.to_display(e_si, Quantity.EFIELD) == pytest.approx(1.0)


def test_entry_round_trip():
    for system in UNIT_SYSTEMS.values():
        for q in (Quantity.CHARGE, Quantity.LENGTH):
            for v in (0.0, 1e-9, -4.2e-6, 7.0):
                assert system.from_entry(system.to_entry(v, q), q) == pytest.approx(v)


@pytest.mark.parametrize("value, expected", [
    (0.0, "0 C"),
    (1e-9, "1 nC"),
    (1.234e-6, "1.23 µC"),
    (-2.5e-11, "-25 pC"),
    (999.0, "999 C"),
    (1500.0, "1.5 kC"),
    (float("inf"), "—"),
    (float("nan"), "—"),
])
def test_prefix_formatting(value, expected):
    assert format_value(value, "C", prefixes=True) == expected


def test_cgs_formatting_uses_scientific_notation_outside_a_sane_range():
    # inside [1e-3, 1e5) plain decimal at 3 significant figures
    assert format_value(29.98, "statC", prefixes=False) == "30 statC"
    assert format_value(0.04217, "statC", prefixes=False) == "0.0422 statC"
    # outside that range, scientific notation -- CGS never uses SI prefixes
    assert format_value(3.0e7, "statV", prefixes=False) == "3.00e+07 statV"
    assert format_value(1.0e-6, "statA", prefixes=False) == "1.00e-06 statA"


def test_fmt_bare_keeps_the_prefix_but_drops_the_symbol():
    # A colorbar tick reads "1n" against a title of "q (C)".
    assert SI.fmt_bare(1e-9, Quantity.CHARGE) == "1n"
    assert SI.fmt_bare(2.2e6, Quantity.EFIELD) == "2.2M"
    assert SI.fmt_bare(298.0, Quantity.POTENTIAL) == "298"
    assert SI.fmt(1e-9, Quantity.CHARGE) == "1 nC"
