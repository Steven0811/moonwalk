# cad/ — Sizing Analysis and Code-Based Modeling

This file covers phase 1 (offline feasibility analysis), phase 2 (build123d modeling), and the printing and weighing parts of phase 4. Motor specs, joint conventions and hand-off rules are in the root [CLAUDE.md](../CLAUDE.md).

**This folder's output**: the STL, STEP, URDF, view images and model report under `cad/out/`. `train/` only reads these files.

---

## Phase 1: Sizing feasibility analysis

Before modeling, prove that a set of dimensions is feasible using offline tools (`cad/analysis/`, plain Python; needs neither Isaac Sim nor build123d).

### Geometry (starting values; the analysis tools set the final ones)

| Parameter | Starting value | Rationale |
| --- | --- | --- |
| Leg length (hip axis to sole) | 0.18 m | Derived from the 0.216 Nm rated load (see the design rules in the root file) |
| Thigh | 0.080 m | — |
| Shank | 0.072 m | Slightly shorter than the thigh, breaking the equal-link symmetry so the knee stays away from the fully-extended singularity in its working range |
| Ankle height | 0.028 m | As low as possible: the dorsiflexion needed for a flat foot is knee − hip, so a lower ankle needs less; but it must still fit a servo or linkage |
| Foot length (ankle to toe / ankle to heel) | 0.036 / 0.023 m | — |
| Foot width, stance width | Set by the static stability analysis | They are the only source of lateral stability (no hip roll by default; see "Consequences of having no hip roll" in the root file) |
| Nominal hip height | 90–95% of leg length | Knees pre-bent 15–30°; see the conflict below |

**A conflict that must be resolved in phase 1**: with the hips high, the knee is nearly straight and the sliding foot cannot reach the floor at the back of the slide. Lowering the hips bends the knee, but then a flat foot needs more ankle dorsiflexion than the joint allows, and knee torque rises. Leg length, ankle range and knee torque are compatible in pairs but not all three at once.

Design requirement: **at the nominal hip height, the sliding foot stays flat on the floor at the back end of the slide, with the knee bent 15–30°, the ankle within its range, and every joint's static torque ≤ 0.216 Nm.**

### Static stability

If the center of pressure stays in front of the center of mass for the whole cycle, the robot falls backward at the open-loop inverted-pendulum time constant. That is a structural problem that control tuning cannot fix. The moonwalk keeps both feet on the ground throughout, so avoid it through sizing:

- In the slow version with a 4-second gait cycle, **the center-of-mass projection must stay inside the support polygon in every phase, with ≥ 1 cm margin both fore-aft and laterally**
- Prove this in this phase with the offline tools, not by trial in simulation
- Gait phases are defined in train/CLAUDE.md; the analysis tools read the same `config/gait.yaml`

### Analysis tools

- `kinematics.py`: forward and inverse kinematics, reachability, flat-foot condition
- `statics.py`: static joint torques, center of mass, support polygon and margins for every phase
- `sizing_report.py`: sweeps leg length, thigh/shank ratio, ankle height, nominal hip height, foot size, stance width, wheel position and wheel diameter, and writes the feasibility report `cad/out/sizing_report.md`

---

## Mass budget

Target total mass ≤ 0.7 kg (estimates; weigh the parts once printed):

| Item | Estimate |
| --- | --- |
| Servos × 6 | 126 g (× 8 = 168 g if hip roll is added) |
| Battery (model not chosen; provisionally a ~450 mAh 2S LiPo) | ~30 g |
| Controller board, bus adapter, IMU (models not chosen) | ~25 g |
| Printed structure | ~300 g |
| Screws, bearings, wiring | ~80 g |
| Foot wheel sets × 2 feet (wheels, axles, bearings, one-way clutches) | ~6–12 g |
| Total | ~570 g, limit 700 g |

- **Keep mass proximal**: battery and controller in the pelvis; mount the knee and ankle servos as high as possible (e.g. the ankle servo at the top of the shank, driving the ankle through a linkage) so the shank and foot stay light. A linkage adds backlash; report the backlash value to `train/` for the servo model.
- Structural material: PETG or PLA+ for load-bearing parts. Record infill and wall thickness in the BOM, since they determine mass.
- Printed part masses start from slicer estimates. **Weigh them once printed, write the values back to `robot.yaml`, and regenerate the model.**

### Electronics (models not chosen yet)

The battery, controller board, bus adapter and IMU have not been chosen. Until they are, handle them with **adjustable placeholders**; do not delay modeling because of them:

- **Parameterized**: the `electronics:` section of `robot.yaml` lists each component's size, mass and mounting position (in the pelvis frame). Use the largest candidate's size plus some margin; use estimated masses for now. Once models are chosen, only this section changes.
- **Placeholder blocks**: each component is a block with mass in the model. Like servos and screws, it counts toward `base_link`'s mass, center of mass and inertia, and takes part in the interference sweep.
- **Separate electronics tray**: a tray screwed to the pelvis, with a generic hole grid or strap slots; components are held with hook-and-loop tape or zip ties. Once models are chosen, only the tray is reprinted, not the pelvis.
- **Fore-aft adjustable battery mount**: the battery is the heaviest single component with the most freedom in placement. Make its mount a fore-aft slide rail (or several hole positions) so the battery can serve as ballast to fine-tune the fore-aft center of mass on the real robot.
- **Reserved space and openings**:
  - USB port (flashing, debugging) facing outward, reachable without disassembly
  - A place for the power switch
  - Battery replaceable without removing screws
  - Routing paths and lengths for the servo bus cables from the tray to each joint
- **Power supply not decided yet**: if a 4.8 V regulated supply is chosen, it needs a regulator able to handle peaks above 7 A, and the tray must leave room for it. Put a switchable `regulator` placeholder in the `electronics:` section for now.

---

## Foot design

### Zones and recess

1. **Which zone slides and which grips is the opposite of intuition**: the toe-raised foot carries the weight, so the **toe pad must grip** (μ about 0.7–0.8); the flat foot must slide, so the **sole must be slippery** (μ about 0.05–0.15).
2. **The toe pad sits higher than the sole (recessed)**, so a flat foot touches the ground only with the sliding part, and the toe pad only touches after about 10° of plantarflexion. Recess depth = toe pad length × tan(10°); e.g. a 15 mm pad is recessed about 2.6 mm.
3. Starting materials for the real robot: PTFE tape on the sole; TPU 85A or silicone for the toe pad. Measure μ with the incline method (μs = tan of the angle at which sliding starts) and write it back to `robot.yaml`.

### One-way foot wheel (primary design; PTFE sole as the control)

The advisor suggested replacing the low-friction material with wheels. **The wheels go under the midfoot to heel, replacing the sliding sole; the toe pad stays a gripping material.**

**Not at the toe.** The toe-raised foot rests only on its toe and carries most of the weight, so it must not move. Wheels at the toe would roll the pivot away.

| Ankle angle | What touches the ground | Fore-aft | Lateral |
| --- | --- | --- | --- |
| Flat | Wheels | Rolls freely backward (the slide); locked forward by the one-way clutch | Wheels grip sideways |
| Plantarflexed past ~10° | Toe pad (wheels lifted) | Grips | Grips |

**Why a one-way clutch**: in the moonwalk, each foot either slides backward relative to the ground or stays still; it never moves forward. Wheels that roll backward and lock forward give a purely mechanical "phase-dependent friction" and keep the support foot from slipping forward. The cost is that the robot cannot move forward without lifting a foot, which does not matter for the moonwalk.

**Design requirements**

- Two wheels per foot, one on each side, on a common axle, so lateral contact is not a single point. Make the track as close to the foot width as possible.
- Starting specs: 8–12 mm wheel diameter, 3–6 mm bore bearings (miniature bearings such as MR63 or 683ZZ), needle-roller one-way clutches (such as HF0612). Actual part numbers depend on the available print space.
- With the foot flat, the wheels sit about 1 mm below the sole so only the wheels touch; past about 10° of plantarflexion the wheels lift and the toe pad touches. Wheel position, wheel diameter and toe pad height together set the switch-over angle; determine them geometrically in phase 1.
- Rolling resistance is about 0.01–0.03, which may be lower than the slide needs and cause overshoot. Leave room for an adjustment mechanism: an O-ring pressed on the wheel or an adjustable friction pad, still targeting an equivalent friction of 0.05–0.15.
- Wheels add distal mass. Keep each foot's wheel set under 6 g, preferably printed rims on metal axles.
- Build the foot in two variants, `wheel` and `ptfe`, switched in `robot.yaml`. Both must be able to produce a URDF.

---

## Phase 2: Code-based modeling (build123d)

Claude Code models the robot in [build123d](https://github.com/gumyr/build123d), a Python code-based CAD library built on OpenCascade (the same kernel as FreeCAD). One program reads dimensions from `config/robot.yaml` and produces, in one run:

- **STL**: for 3D printing, one file per printed part, already rotated to the suggested print orientation
- **STEP**: for viewing and measuring in FreeCAD or Fusion 360
- **URDF**: for `train/` to import into Isaac Lab; masses, inertias and collision shapes are computed by the program, with no exporter in between
- **Model report**: mass, center of mass and inertia per link, interference check results, and view images

The user views the 3D model live in VS Code with the OCP CAD Viewer extension (`ocp_vscode`).

Exporting a model with a CAD tool's URDF exporter commonly causes four problems. Each one produces a simulation that looks fine but is wrong. Code-based modeling avoids them at the root:

| Common problem | Consequence | What this project does |
| --- | --- | --- |
| Parts keep a default material (e.g. steel), making the feet over 40% of total mass | Mass distribution is completely wrong | Every solid must carry a material tag; a missing tag is an error |
| Joints keep exporter defaults: `effort=100`, `velocity=100`, knee ±45° symmetric | The knee can bend backward; torque and speed are unlimited | URDF limits are written directly from `robot.yaml` |
| Left and right legs drawn separately, joint offsets not mirrored | The legs are asymmetric | The right leg is always generated by mirroring the left |
| Visual STL used as the collision body | Collision becomes a convex hull and recesses are filled in | Collision bodies are always primitive shapes |

### 1. Environment

- Create a separate conda environment (e.g. `cad`) and install `build123d` and `ocp_vscode`. **Do not install them into Isaac Sim's bundled Python (Isaac Lab lives there) or into the base environment.**
- Pin package versions in `cad/requirements.txt`.

### 2. Parameters and coordinates

- **All dimensions are read from `robot.yaml`** (loaded by `cad/params.py`); no hard-coded numbers in part code. Print-related values (wall thickness, fillets, fit tolerances, heat-set insert hole sizes, build volume) go in the `print:` section of `robot.yaml`.
- **Units**: build123d works in mm; URDF uses m, kg and kg·m². **Convert in exactly one place, where the URDF is generated**, and verify with unit tests (e.g. a 10 mm PLA cube = 1.24 g, inertia about a central axis = m·a²/6).
- **Coordinates and zero pose** follow the shared conventions in the root file.
- **Each link is modeled in its own frame**, with the origin on the joint axis connecting it to its parent, matching URDF convention. Assembly places links using only joint transforms, and URDF generation uses the same joint transforms, so the two cannot disagree.

### 3. Parts, links and joints

- **One function per part** (`cad/parts/`), returning a build123d solid with a material tag. `cad/links.py` assembles parts into links.
- **Horn interface**: the output shaft is a 25T spline, 4.95 mm OD. Printed parts do not reproduce the spline; they use the stock horn fastened with screws, with a locating recess for the horn in the printed part.
- **Names, joint axes and positive directions** follow the shared conventions in the root file and are read from `robot.yaml`.
- **The right leg is generated by mirroring the left** (about the YZ plane); no separate code. Pitch joints about X keep their direction when mirrored; if hip roll is added, roll joints about Y reverse. The axis signs for each side are written from `robot.yaml` and verified by the checks.
- **Make the hip a replaceable module**: by default the hip-pitch servo mounts directly to the pelvis, but its servo mount is a separate printed part screwed to the pelvis. To add hip roll later, only this module is replaced and the `L_hip` link added; the pelvis and legs are not redone. Leave space and mounting holes for hip-roll servos on both sides of the pelvis.
- **Hip roll switch**: with the switch in `robot.yaml` off, no `L_hip` link or `L_hip_roll` joint is generated; with it on, both are. Both variants must pass `cad/checks.py`.

### Servo model (vendor STEP)

The vendor STEP file lives at `cad/vendor/feetech_hd1910/HD-1910-C001.step`, with its source and download date recorded in `SOURCE.md` in the same folder. **Never modify the original file**; all transformations are done in code in `cad/parts/servo.py`.

**Validate it right after import**. If any check fails, stop and report it; do not fix it on your own:

- Units are mm: the bounding box should be 34 × 20 × 23 mm (within 0.3 mm)
- It is a valid closed solid (volume > 0, valid geometry). If not, use the STEP for visuals only and compute mass and inertia from the simplified shape

**Define the servo's local frame**: origin at the intersection of the output-shaft centerline and the horn mounting face, output shaft along +Z. The transform from the STEP's original placement into this frame lives in `servo.py`; confirm the output-shaft position in the view images.

**Measure key dimensions from the STEP in code**, not by eye:

- Output shaft position and horn mounting face height
- Positions and diameters of the mounting holes (or mounting ears)
- Cable exit position

Write the measurements to the `servo.geometry` section of `robot.yaml` and list them in the model report.

**Use different geometry for different purposes**:

| Purpose | Geometry used | Reason |
| --- | --- | --- |
| Visuals, assembled STEP output, URDF visual | Full vendor STEP | Correct appearance |
| Mounting pockets in printed parts (boolean subtraction) | Simplified shape built from the measured dimensions + fit clearance | Labels, engraving and other details on the vendor model must not be cut into printed parts |
| Mass, center of mass, inertia | Mass fixed at 21 g; center of mass and inertia from the simplified shape | The STEP may be a hollow shell or contain unknown parts, so its volume cannot be trusted |
| Interference sweep | Simplified shape | The full STEP is too detailed and makes the sweep slow |

List the bounding-box difference between the simplified shape and the full STEP in the model report (< 0.5 mm in every direction).

### 4. Mass and inertia

- **Every solid must have a material tag**; a missing tag is an error, and there are no defaults.
- **Printed parts**: the material tag is an "effective density". Start from an estimate based on infill and wall thickness; once the slicer gives the actual mass, write the effective density back to `robot.yaml`. Solid PLA is about 1.24 g/cm³; printed parts are noticeably lower.
- **Purchased parts** (servos, battery, controller board, IMU, bearings, one-way clutches, screws, heat-set inserts): modeled as simplified shapes with mass assigned directly (from spec or weighing), and density derived as mass ÷ volume. Screws and inserts are light individually but total about 80 g; they must be placed on the correct links, not omitted.
- **Center of mass and inertia**: use `GProp` from OCP (build123d's underlying layer) to compute each solid's volume, center of mass and inertia tensor, multiply by density, then combine into the link's center-of-mass frame with the parallel-axis theorem.
- **Checks**:
  - Compare each link's inertia with the box formula for the same size and mass; a difference of an order of magnitude or more means a unit error
  - Compare total mass with the mass budget
  - Compare center-of-mass height with the phase 1 report; if it differs by more than 10%, go back to phase 1 and recompute
- **Weigh parts once printed**, write each part's actual mass into `robot.yaml` to override the estimate, and regenerate the URDF.

### 5. Collision and visual geometry

- **URDF collisions are always primitives** (box, cylinder, sphere), generated from the collision-shape definitions in `robot.yaml`, never from STL.
- **Split each foot into at least three separate collision shapes**: toe pad, sole (or wheels), heel. Wheels are spheres (a ~1 cm-diameter cylinder is approximated as a convex hull in PhysX and tends to jitter).
- URDF cannot express PhysX friction materials, so every collision element must be named (e.g. `L_foot_toe_pad`); `train/`'s post-processing script binds physics materials by name. **If the naming scheme changes, tell `train/`.**
- Make leg collision bodies slightly smaller than the outer shape so adjacent links do not overlap at the joints.
- **URDF visuals use STL**, one merged file per link, with a coarse mesh; they are for display only.

### 6. Mechanism details

- **Support each servo output with a bearing on the opposite side** (dual support) so the servo shaft carries no cantilever load. This also reduces backlash.
- **Hard stops are built into the printed parts**, at angles equal to the joint ranges in `robot.yaml`. The servo itself can turn 360°, so the hard stops are the only limits.
- **The toe pad (TPU) is a separate part** belonging to the `L_foot` link, with its own collision shape and material.
- **Cables**: servo cables are 15 cm. Leave enough slack for every joint's full range, and route cables inside the links or close to the joint axes.
- **Printing**: do not let the main bending loads pull layers apart. Use heat-set inserts (M2/M2.5) for load-bearing holes. Fit tolerances (bearing press fits, servo pockets, insert holes) are `print:` parameters. Claude Code first generates a **tolerance test piece**; the user prints it, reports which sizes fit, and the values are written back.

### 7. Automated checks

`cad/checks.py` runs after every modeling change. If any check fails, no URDF is written:

- **Interference sweep**: sample each joint over its full range (at least every 5°), plus combinations with both legs at their limits, and compute the intersection volume between non-adjacent links; it must be 0. Between adjacent links, only the designed shaft and bearing overlaps are allowed.
- **Hard stop positions**: the angle at which each hard stop engages equals the joint range in `robot.yaml` (within 1°).
- **Build volume**: every printed part fits the printer's build volume (`print.bed_size`).
- **Left-right symmetry**: masses, centers of mass and joint origins of left and right links must be mirror images (within 0.5 mm).
- **Zero pose**: with all joints at 0°, both soles are parallel to the ground and at the same height (within 0.5 mm).
- **Foot switch-over angle**: with the foot flat, only the sole or wheels touch the ground; at the switch-over angle (8–12° of plantarflexion), the toe pad touches instead.
- **Electronics placeholders**: every placeholder block fits in the tray without interfering with the others, and the battery stays inside the allowed placement region from the phase 1 report over the full travel of its rail.

### 8. Outputs and visual review

`python cad/build.py` produces in one run:

| Output | Location | Purpose |
| --- | --- | --- |
| Printed-part STL | `cad/out/stl/` | Slicing and printing |
| Assembly and part STEP | `cad/out/step/` | Viewing in FreeCAD or Fusion |
| URDF and visual meshes | `cad/out/urdf/` | Hand-off to `train/` |
| Front, side and top views plus an isometric PNG | `cad/out/views/` | Visual review |
| Model report | `cad/out/model_report.md` | Mass, center of mass and inertia per link; total mass; center-of-mass height; check results |

- **After every model change, Claude Code opens the view PNGs and looks at them itself** to confirm the shape is as expected, before handing off to the user. Passing numeric checks does not mean the shape is right.
- The user reviews the 3D model in OCP CAD Viewer in VS Code and gives feedback. Claude Code changes the code accordingly and never edits output files by hand.
- `cad/out/` is generated: never edit it by hand; change the code or `robot.yaml` and regenerate.

---

## File structure

```
cad/
  CLAUDE.md
  requirements.txt
  analysis/             # Phase 1
    kinematics.py
    statics.py
    sizing_report.py
  vendor/               # Vendor files; do not modify
    feetech_hd1910/
      HD-1910-C001.step
      SOURCE.md         # Source URL, download date
  params.py             # Reads config/robot.yaml
  parts/                # One function per part: pelvis, thigh, shank, foot, toe pad, wheel, tolerance test piece…
    servo.py            # Imports the vendor STEP; validation, local frame, key dimensions, simplified shape
  links.py              # Parts → links, with materials
  assembly.py           # Assembles the zero pose using joint transforms
  mass.py               # Mass, center of mass, inertia
  urdf.py               # Generates the URDF (primitive collisions, limits)
  checks.py             # Interference sweep, hard stops, build volume, symmetry, zero pose, foot switch-over angle
  build.py              # Produces all outputs in one run
  tests/                # Unit tests for unit conversion and mass computation
  out/                  # Generated; never edit by hand: sizing_report.md, stl/, step/, urdf/, views/, model_report.md
```

---

## Task breakdown

**Phase 1: Offline feasibility**
1. Create `config/robot.yaml`, `config/gait.yaml` and `cad/analysis/`
2. Write `kinematics.py` and `statics.py`: over the full gait cycle, compute reachability, required ankle angles, each joint's static torque, and the center-of-mass margins to the support polygon
3. Sweep the dimensional parameters, find the feasible region, and write the feasibility report. The sweep also covers the electronics uncertainty: total mass ±50% (about 25–80 g) and the battery's fore-aft and vertical position in the pelvis. The report must list the **battery's allowed placement region**: anywhere inside it, every phase keeps a static margin ≥ 1 cm
4. Wheel geometry: from the ankle height and foot size, set the wheel's fore-aft position, wheel diameter and toe pad height so the switch-over angle is about 10°, and confirm the wheels do not interfere with servos or linkages
5. Pass criteria: a parameter set exists that satisfies the torque budget (static p99 ≤ 0.216 Nm), a slow-gait static margin ≥ 1 cm, a sliding foot that reaches the floor throughout, a switch-over angle of 8–12°, and a battery placement region that is non-empty and fits inside the pelvis. If none exists, report which constraint blocks it; the user decides which one to relax

**Phase 2: Code-based modeling**
6. Create the `cad` conda environment and the `cad/` skeleton; write the unit tests for unit conversion and mass computation first
7. Import `cad/vendor/feetech_hd1910/HD-1910-C001.step` and complete the validation, local frame, key-dimension measurement and simplified shape (see "Servo model (vendor STEP)"). Then generate the tolerance test piece STL (including a servo pocket) for the user to print, and write the results back to the `print:` parameters
8. Build all parts and links from the feasibility report's dimensions, generating the right leg by mirroring; build both the `wheel` and `ptfe` foot variants; model electronics as placeholder blocks on a separate tray (see "Electronics (models not chosen yet)")
9. Run `cad/build.py` to produce all outputs. Claude Code reviews the view images first, then hands off to the user for review in OCP CAD Viewer, and revises based on feedback
10. Pass criteria: `cad/checks.py` passes in full; total mass and center-of-mass height are within 10% of the phase 1 report; the user approves the shape. Once passed, announce that `train/` can start phase 3

**Phase 4 (printing and weighing)**
11. Slice the STL from `cad/out/stl/` and fill in `hardware/bom.md` (purchased parts; printed parts' material, infill and slicer mass)
12. Once parts are printed, weigh them and write the actual masses and measured μ back to `robot.yaml`; rerun `cad/build.py` and tell `train/` to re-import and rerun acceptance
13. Once electronics models are chosen: update the `electronics:` section (actual dimensions, weighed masses, mounting positions), reprint the tray, rerun `cad/checks.py` and the phase 1 static analysis, then tell `train/` to replace the randomization center values with the actual ones
