# emsim

An interactive 3-D electromagnetics sandbox. Place point charges in space, drag
them around, and watch the Coulomb force, the electric field, and the electric
potential update live. Units switch between SI and Gaussian CGS.

Built on PyQt6 + pyqtgraph's OpenGL viewport, in a `uv`-managed venv.

## Running

```sh
uv run main.py          # or: uv run emsim
uv run pytest           # 63 tests, no display required
```

`uv sync` provisions Python 3.13 and the dependencies on first run.

## Controls

| Action | Input |
| --- | --- |
| Orbit / pan / zoom | drag empty space · middle-drag · wheel |
| Select a charge | click it |
| Move it in the view plane | drag it |
| Move it along **z** | shift-drag it |
| Add a charge | double-click empty space, `Add`, or `Ctrl+N` |
| Delete the selection | `Del` |
| Reset the camera | `Ctrl+R` |

Charge magnitude and position are also editable in the table on the right, in
whichever unit system is active.

## The three visualizations

Each is an independent checkbox; any combination can be on at once.

**Force** — one arrow per charge, starting at the sphere's surface, labelled
with its magnitude.

Length is proportional to |**F**| on a **fixed, absolute scale** — it is *not*
renormalised to the largest force on screen. That is deliberate: it means the
arrows genuinely lengthen as two charges approach and shorten as they separate,
so the 1/*r*² falloff is something you watch rather than something you read off
the labels. Halving a separation quadruples the arrow.

At gain 1×, an arrow of 0.30 × the domain half-width represents the force
between two 10 nC charges 0.7 m apart — the dipole the app opens with. The
gain slider is logarithmic and spans 10⁻⁶ to 10⁺⁶, since an absolute scale
needs real range when you work at a different length or charge scale. The
**Fit** button next to it sets the gain *once* so the largest force currently
present draws at the reference length; the scale stays fixed afterwards.

Arrow length is capped at 8 × the half-width, far outside the visible box, so
it never truncates a force you are looking at — it only stops near-coincident
charges from generating absurd geometry.

**Electric field** — a fixed-length arrow at every point of an *n*³ grid,
coloured by |**E**| against a colorbar. Length is deliberately constant so that
colour is the only magnitude channel; a log scale is the default because a
Coulomb field spans decades across any interesting region. Samples that land
inside a charge are dropped rather than drawn.

Shading is baked into the vertex colours with a clamped ambient term rather
than left to the GL `shaded` shader. Full Lambertian shading swings a face's
brightness across the entire colormap, which would make a glyph's colour
disagree with the colorbar it is supposed to be read against.

**Potential** — a semi-transparent heatmap on a diverging colormap, as either:

- *Volume*: the whole box, with opacity rising from zero where *V* ≈ 0. The
  render composites ~*n* stacked slices, so per-voxel alpha is set to
  `1 - (1 - opacity)^(1/n)`, making the slider mean "opacity of a full traverse"
  instead of something that saturates the moment resolution changes.
- *Slice plane*: one cut plane at a chosen x, y, or z.

Colour uses a symmetric-log scale by default (a Coulomb potential has no useful
dynamic range on a linear one), but opacity always ramps linearly in |*V*| — a
log ramp would leave the entire box hazy, since the far field is not small on a
log scale.

## Units

The core computes in SI only; unit systems are a display-layer concern
(`emsim/units.py`). Each system maps a `Quantity` to a display unit and a
factor. SI uses engineering prefixes (`12.3 nC`); Gaussian CGS uses plain or
scientific notation, as is conventional.

`Quantity` already covers magnetic field, current, velocity and energy, so the
unit machinery needs no changes when those quantities start being displayed.
Tests assert that Gaussian conversions are self-consistent — that **F** = *q***E**
and *V* = *E·d* still hold after conversion, and that 1 statC at 1 cm gives
exactly 1 statV.

## Physics notes

Charges are modelled as **uniformly charged balls**, not true points. Outside
the radius this is exactly Coulomb's law; inside, the field and potential
follow the solid-sphere solution (**E** → 0 at the centre, *V* finite). This
removes the 1/*r*² singularity without any unphysical fudging, and the ball's
radius is exactly what gets drawn, so there is no discrepancy between the
picture and the model. Radius scales as |*q*|^(1/3) at constant charge density.

## Architecture

```
emsim/
  units.py          UnitSystem, Quantity, formatting        (no Qt, no numpy deps)
  core/
    entities.py     PointCharge (carries velocity + mass, unused by statics)
    solvers.py      Solver protocol; CoulombSolver -> {"E", "V"}
    scene.py        entities + solvers + observers          (no Qt)
  viz/
    colormaps.py    self-contained LUTs
    norms.py        Linear / Log / SymLog, each generating its own ticks
    geometry.py     vectorised arrow-mesh construction
    layers.py       Charge / Force / EField / Potential layers + LayerStack
    view3d.py       picking, dragging, camera maths
    overlay.py      2-D label layer stacked over the viewport
    colorbar.py     QPainter colorbar driven by a ColorScale
  ui/
    controls.py     right-hand panel
    main_window.py  wiring + coalesced redraw loop
```

Two seams carry the extensibility:

**Solvers.** A solver is anything with `evaluate(scene, points, exclude_uid)`
returning a dict of named field arrays. `Scene.evaluate` sums the contributions
of every registered solver, so fields superpose automatically.

**Layers.** A layer owns some GL items, rebuilds them from the scene, and may
publish a `ColorScale` (which the UI renders as a colorbar) and `Label3D`s
(which the overlay draws).

### Adding the next piece of physics

*Magnetostatics.* Write a `BiotSavartSolver` with `provides = ("B", "A")` and
append it to `Scene.solvers`. Add a `BFieldLayer` — it can subclass
`EFieldLayer` almost verbatim, swapping the `"E"` key for `"B"` and the
quantity for `Quantity.BFIELD`, which already has SI and Gauss units defined.
Register it in `LayerStack` and add a checkbox.

*Moving charges.* `PointCharge.velocity` and `Scene.time` already exist. Add an
integrator that advances positions from `Scene.forces()`, drive it from a
`QTimer` in `MainWindow`, and call the existing `request_redraw()`. The redraw
path is already coalesced and drops resolution during interaction, so it will
keep up.

*Radiation.* A `LienardWiechertSolver` needs each charge's retarded state,
which means `PointCharge` must keep a trajectory history rather than a single
position — the one change to the entity model that the current design does not
already absorb. Everything downstream (superposition, layers, units, colorbars)
works unchanged.

## Performance

Rebuilding force + a 9³ field + a 40³ potential volume takes ~15 ms; during a
drag, resolution drops to 60 % and it takes ~4 ms. Redraws are coalesced on a
16 ms timer. A 64³ volume costs ~135 ms, which is why the drag-quality
reduction exists.
