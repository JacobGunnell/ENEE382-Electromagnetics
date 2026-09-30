# emsim

An interactive 3-D electromagnetics sandbox. Place point charges and charged
bodies in space, drag them around, and watch the Coulomb force, the electric
field, and the electric potential update live. Conductors solve for their own
equilibrium charge distribution, so you can measure voltage and capacitance
between any two of them. Units switch between SI and Gaussian CGS.

Built on PyQt6 + pyqtgraph's OpenGL viewport, in a `uv`-managed venv.

## Running

```sh
uv run main.py          # or: uv run emsim
uv run pytest           # 120 tests, no display required
```

`uv sync` provisions Python 3.13 and the dependencies on first run.

## Controls

Three tools, picked at the top of the panel.

**Select** (default)

| Action | Input |
| --- | --- |
| Orbit / pan / zoom | drag empty space · middle-drag · wheel |
| Select a charge or body | click it |
| Move it in the view plane | drag it |
| Move it along **z** | shift-drag it |
| Add a point charge | double-click empty space, or `Ctrl+N` |
| Delete the selection | `Del` |
| Reset the camera | `Ctrl+R` |

**Add charge** — click a body to deposit the amount in the spin box onto it,
spread uniformly. Clicking again adds more. On an insulator the charge stays
where it lands; on a conductor it migrates to the surface (see below).

**Measure** — click two conductors to read the voltage between them and the
capacitance of the pair.

Everything is also editable numerically in the tables on the right, in whichever
unit system is active. **Hovering** a body makes it glassy so you can see the
charge inside it.

## Bodies

Five shapes — **line**, **loop**, **sheet**, **disk**, **sphere** — each either a
**conductor** or an **insulator**.

An insulator holds whatever charge you put on it, spread uniformly: through the
volume for a sphere, over the area for a sheet or disk, along the wire for a
line or loop. Hover it and the dots inside become visible.

A conductor cannot do that. Charge on a conductor arranges itself so the surface
is an equipotential, which means it ends up on the surface and piles up wherever
that surface curves sharply. Deposit charge on one and you watch it happen: the
body turns glassy, the dots stream out of the interior, and they settle into the
solved distribution. The field and potential follow the charge as it moves. That
animation is a relaxation sequence for the eye, not time-accurate dynamics —
real charge settles in femtoseconds.

Dot area tracks the charge on each site, so the pile-up at a plate's rim is
something you see rather than something you have to infer.

### What is *not* modelled

"Insulator" here means *charge stays where you put it*. It does **not** mean a
dielectric medium: there is no relative permittivity, no bound surface charge,
no polarisation response to an applied field, and no ε_r factor anywhere. Every
body sits in vacuum. Putting a slab between two plates will therefore not raise
their capacitance the way a real dielectric would. Adding that means solving for
polarisation alongside the conductor charges, which is a bigger change than
anything described under *Adding the next piece of physics*.

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

## Voltage and capacitance

Pick **Measure** and click two conductors. The panel reports each one's
potential, the difference, each one's charge, and:

* **C** — the capacitance of the pair, `1 / (P_aa + P_bb - 2 P_ab)` from the
  conductor potential-coefficient matrix. This is the geometry's capacitance:
  what you would measure with +Q on one and -Q on the other. It does not depend
  on what charge happens to be on them at the moment, so you can watch it change
  as you drag the plates together while the charge stays put.
* **Q/ΔV** — the same number computed the naive way. It agrees with C exactly
  when the two carry equal and opposite charge, and the panel says so when they
  do not, because then Q/ΔV is not a capacitance at all.
* **½CΔV²** — the stored energy.

Any other conductors in the scene are included as floating uncharged bodies,
which is what they physically are — so a third plate nearby changes the answer,
correctly.

In Gaussian units capacitance is a **length**, in centimetres: an isolated
sphere of radius R has C = R exactly. The app reports it that way.

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

**Point charges** are modelled as uniformly charged balls, not true points.
Outside the radius this is exactly Coulomb's law; inside, the field and
potential follow the solid-sphere solution (**E** → 0 at the centre, *V*
finite). This removes the 1/*r*² singularity without any unphysical fudging, and
the ball's radius is exactly what gets drawn, so there is no discrepancy between
the picture and the model. Radius scales as |*q*|^(1/3) at constant density.

**Bodies** are discretised into equal-measure sites, and every field is an
explicit sum over those sites. There is no infinite-sheet or infinite-wire
idealisation anywhere in the program, so fringing at a plate's edge is simply
what the sum gives.

**Conductors** are solved by a boundary element method. With one unknown charge
per site and one unknown potential per conductor, the equilibrium conditions
become one linear system:

    [ P   -B ] [q]   [ -V_ext ]
    [ B^T  0 ] [V] = [   Q    ]

The first block row says every site of a conductor is at that conductor's
potential; the second says charge is conserved on each body. Off-diagonal
entries of **P** are `k / |r_i - r_j|`; the diagonal is the site's own
potential — an equal-area disc for a surface patch, a segment of wire for a line
element, using `asinh` rather than the usual `log` so it stays valid when the
segments get shorter than the wire is thick.

Inverting once buys more than a solve. In block form the inverse's lower-right
corner *is* the conductor potential-coefficient matrix, so the capacitance
matrix costs nothing extra, and re-solving while a point charge is dragged past
a plate is one matrix-vector product rather than a re-inversion.

### Validation

The solver is checked against results that hold in closed form, including ones
that exist *only* because of edge effects — a solver that quietly assumed
uniform density could not pass these:

| Check | Result |
| --- | --- |
| Isolated sphere, C = 4πε₀R | converges, <0.5% at 1500 sites |
| Isolated **disc**, C = 8ε₀R | converges, <0.5% at 2000 sites |
| Disc charge density ∝ 1/√(R²−r²) | holds to 6%; rim is 5.4× the centre |
| Field inside a conductor | <1% of the external field |
| Conductor surface | equipotential to <1% |
| Force on a charged sphere from a point charge | matches the image-charge solution to 0.5% |
| Point charge near a plate | attracted, matches the image force |
| Neutral conductor near a charge | net zero, near face opposite in sign |
| Parallel plates | C exceeds ε₀A/d, and the excess shrinks as the gap closes |
| Uniformly charged insulating ball | reproduces Gauss's law inside and out |

## Architecture

```
emsim/
  units.py          UnitSystem, Quantity, formatting        (no Qt, no numpy deps)
  core/
    entities.py     PointCharge (carries velocity + mass, unused by statics)
    bodies.py       Line/Loop/Sheet/Disk/Sphere: sampling, ray casting, meshes
    kernels.py      the chunked Coulomb sum; everything goes through it
    electrostatics.py  conductor equilibrium (BEM) + capacitance
    solvers.py      Solver protocol; CoulombSolver -> {"E", "V"}
    scene.py        entities + bodies + solvers + observers (no Qt)
  viz/
    colormaps.py    self-contained LUTs
    norms.py        Linear / Log / SymLog, each generating its own ticks
    geometry.py     vectorised arrow-mesh construction
    layers.py       Charge / Cloud / Force / EField / Body / Measure /
                    Potential layers + LayerStack
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

*Dielectrics.* The one thing the current design does **not** absorb cheaply,
and the most natural next step given the capacitance work: see *What is not
modelled* above.

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
16 ms timer. A 64³ volume costs ~135 ms, which is why the drag-quality reduction
exists.

With two 320-site conductors plus the field and force layers, a redraw is ~26 ms
as long as the conductor geometry has not changed, because the factorisation is
cached and only the right-hand side is re-solved. Moving a *body* does invalidate
it — that costs ~50 ms at full resolution, so sites are coarsened to 45 % for the
duration of the drag and restored on release. The site count per body is
adjustable; cost of the solve grows as the cube of the total, and the total is
capped at 2400.
