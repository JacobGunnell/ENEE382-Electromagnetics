"""Unit systems and quantity formatting.

The simulation core *always* works in SI.  Unit systems are a presentation
concern: they define, for each physical quantity, a display unit and the
factor that converts an SI value into it.  This keeps the physics free of
unit-system branching and makes adding new quantities (magnetic field,
vector potential, Poynting flux, ...) a one-line change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# --------------------------------------------------------------------------
# Physical constants (SI)
# --------------------------------------------------------------------------
C_LIGHT = 2.99792458e8          # m/s
EPS0 = 8.8541878128e-12         # F/m
MU0 = 1.25663706212e-6          # H/m
K_COULOMB = 1.0 / (4.0 * math.pi * EPS0)   # 8.9875517873681764e9 N m^2 / C^2
ELEMENTARY_CHARGE = 1.602176634e-19        # C

# Gaussian-CGS conversion helpers -----------------------------------------
_C_NUM = 2.99792458e9           # 1 C = 2.99792458e9 statC


class Quantity(str, Enum):
    """Kinds of physical quantity the UI knows how to display."""

    LENGTH = "length"
    CHARGE = "charge"
    EFIELD = "efield"
    POTENTIAL = "potential"
    FORCE = "force"
    ENERGY = "energy"
    # Present so that future magnetostatic / radiation layers need no changes
    # to the unit machinery.
    BFIELD = "bfield"
    CURRENT = "current"
    TIME = "time"
    VELOCITY = "velocity"


@dataclass(frozen=True)
class Unit:
    """A display unit: ``value_in_unit = value_si * per_si``."""

    symbol: str
    per_si: float = 1.0


@dataclass(frozen=True)
class UnitSystem:
    name: str
    #: Base unit per quantity, used with engineering (SI) prefixes when
    #: ``use_prefixes`` is set.
    display: dict[Quantity, Unit]
    #: Fixed unit used in numeric entry fields / table headers, where a
    #: shifting prefix would be confusing.
    entry: dict[Quantity, Unit]
    use_prefixes: bool

    def to_display(self, value_si: float, q: Quantity) -> float:
        return value_si * self.display[q].per_si

    def to_entry(self, value_si: float, q: Quantity) -> float:
        return value_si * self.entry[q].per_si

    def from_entry(self, value: float, q: Quantity) -> float:
        return value / self.entry[q].per_si

    def entry_symbol(self, q: Quantity) -> str:
        return self.entry[q].symbol

    def fmt(self, value_si: float, q: Quantity, sig: int = 3) -> str:
        """Format an SI value as a string in this system, with units."""
        unit = self.display[q]
        return format_value(value_si * unit.per_si, unit.symbol,
                            prefixes=self.use_prefixes, sig=sig)

    def fmt_bare(self, value_si: float, q: Quantity, sig: int = 3) -> str:
        """Like :meth:`fmt` but without the unit symbol, for tick labels.

        Any SI prefix is kept and closed up against the mantissa ("1.2M"), so
        a tick reads correctly against the unit in the colorbar title.
        """
        unit = self.display[q]
        text = format_value(value_si * unit.per_si, "",
                            prefixes=self.use_prefixes, sig=sig)
        return text.replace(" ", "")

    def unit_symbol(self, q: Quantity) -> str:
        return self.display[q].symbol


SI = UnitSystem(
    name="SI",
    display={
        Quantity.LENGTH: Unit("m"),
        Quantity.CHARGE: Unit("C"),
        Quantity.EFIELD: Unit("V/m"),
        Quantity.POTENTIAL: Unit("V"),
        Quantity.FORCE: Unit("N"),
        Quantity.ENERGY: Unit("J"),
        Quantity.BFIELD: Unit("T"),
        Quantity.CURRENT: Unit("A"),
        Quantity.TIME: Unit("s"),
        Quantity.VELOCITY: Unit("m/s"),
    },
    entry={
        Quantity.LENGTH: Unit("m", 1.0),
        Quantity.CHARGE: Unit("nC", 1e9),
        Quantity.EFIELD: Unit("V/m", 1.0),
        Quantity.POTENTIAL: Unit("V", 1.0),
        Quantity.FORCE: Unit("N", 1.0),
        Quantity.ENERGY: Unit("J", 1.0),
        Quantity.BFIELD: Unit("T", 1.0),
        Quantity.CURRENT: Unit("A", 1.0),
        Quantity.TIME: Unit("s", 1.0),
        Quantity.VELOCITY: Unit("m/s", 1.0),
    },
    use_prefixes=True,
)

# Gaussian CGS.  Chosen over ESU/EMU because it is the convention used by
# Griffiths/Jackson for the electrostatics + radiation material this tool
# targets.  The two conversions worth spelling out:
#     1 statV/cm = 2.99792458e4 V/m
#     1 statV    = 2.99792458e2 V
_GAUSSIAN_UNITS = {
    Quantity.LENGTH: Unit("cm", 1e2),
    Quantity.CHARGE: Unit("statC", _C_NUM),
    Quantity.EFIELD: Unit("statV/cm", 1.0 / (_C_NUM * 1e-5)),
    Quantity.POTENTIAL: Unit("statV", 1.0 / (_C_NUM * 1e-7)),
    Quantity.FORCE: Unit("dyn", 1e5),
    Quantity.ENERGY: Unit("erg", 1e7),
    Quantity.BFIELD: Unit("G", 1e4),
    Quantity.CURRENT: Unit("statA", _C_NUM),
    Quantity.TIME: Unit("s", 1.0),
    Quantity.VELOCITY: Unit("cm/s", 1e2),
}

GAUSSIAN = UnitSystem(
    name="CGS (Gaussian)",
    display=dict(_GAUSSIAN_UNITS),
    entry=dict(_GAUSSIAN_UNITS),
    use_prefixes=False,
)

UNIT_SYSTEMS: dict[str, UnitSystem] = {SI.name: SI, GAUSSIAN.name: GAUSSIAN}


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
_PREFIXES = [
    (1e24, "Y"), (1e21, "Z"), (1e18, "E"), (1e15, "P"), (1e12, "T"),
    (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"),
    (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f"),
    (1e-18, "a"), (1e-21, "z"), (1e-24, "y"),
]


def format_value(value: float, symbol: str, prefixes: bool = True,
                 sig: int = 3) -> str:
    """Render ``value`` with ``sig`` significant figures and a unit symbol.

    With ``prefixes`` the mantissa is kept in [1, 1000) using SI prefixes
    (``12.3 nC``); otherwise scientific notation is used outside a sensible
    range (``3.45e+07 statC``), which is the convention in Gaussian units.
    """
    if value is None or not math.isfinite(value):
        return "—"
    if value == 0.0:
        return f"0 {symbol}".strip()

    if prefixes:
        mag = abs(value)
        for scale, pre in _PREFIXES:
            if mag >= scale * 0.999999:
                break
        else:
            scale, pre = _PREFIXES[-1]
        scaled = value / scale
        return f"{_sigfig(scaled, sig)} {pre}{symbol}".strip()

    mag = abs(value)
    if 1e-3 <= mag < 1e5:
        return f"{_sigfig(value, sig)} {symbol}".strip()
    return f"{value:.{max(sig - 1, 1)}e} {symbol}".strip()


def _sigfig(x: float, sig: int) -> str:
    if x == 0:
        return "0"
    digits = max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))
    digits = min(digits, 6)
    s = f"{x:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"
