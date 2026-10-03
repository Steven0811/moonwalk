# Moonwalk Mini — A Moonwalking Biped Driven by Feetech HD-1910 Servos

As of: 2026-10-01

This file is the **shared spec**: goals, motor, joint conventions, and the hand-off between modeling and training. Working details live in two sub-files:

| File | Contents | Phases covered |
| --- | --- | --- |
| [cad/CLAUDE.md](cad/CLAUDE.md) | Sizing feasibility analysis, mechanism and mass budget, foot design, build123d code-based modeling, printing | Phases 1 and 2; printing and weighing in phase 4 |
| [train/CLAUDE.md](train/CLAUDE.md) | Isaac Sim import, servo model, gait, acceptance tests, reinforcement learning | Phase 3; calibration and robustness in phase 4 |

When working under `cad/` or `train/`, Claude Code reads both this file and that folder's CLAUDE.md. Anything that applies to both lives only in this file.

---

## Goal and scope

Design a small biped that **can actually be built**, validate it in Isaac Sim, then build it from 3D-printed parts and Feetech HD-1910-C001 servos, and have it perform Michael Jackson's moonwalk.

- At most 4 active DOF per leg, **3 by default** (hip pitch, knee, ankle pitch). Hip roll is added only when one of the "Conditions for adding hip roll" is met. Every joint uses an HD-1910-C001. Apart from motors, electronics, screws and bearings, all structural parts are 3D printed
- The robot's size and weight are derived from what the motor can do, not the other way round
- Four phases: (1) offline sizing and feasibility → (2) code-based modeling in build123d by Claude Code → (3) import into Isaac Sim, validate the gait and train → (4) prepare sim-to-real
- The STL for printing and the URDF for Isaac Sim come from the same modeling program. Do not assemble a separate robot from simple primitives
- Firmware and physical assembly come after phase 4, but from phase 1 onward every design decision must be buildable
- Every phase has pass criteria. Do not move to the next phase until they pass; if they cannot pass, report the measured reason instead of forcing it

The "Hard rules" in each file are mandatory, not reference material.

---

## Motor: Feetech HD-1910-C001

Official specs ([Feetech product page](https://www.feetechrc.com/510257)). **Torque, speed and current are all rated at 4.8 V**:

| Item | Official spec | Converted / notes |
| --- | --- | --- |
| Voltage | 4–8.4 V | Can run directly from a 2S LiPo |
| Stall torque | 9 kg·cm @ 4.8V | **0.88 Nm** |
| Rated load | 2.2 kg·cm @ 4.8V | **0.216 Nm** — the ceiling for sustained load |
| No-load speed | 0.137 s/60° (73 RPM) @ 4.8V | **438°/s = 7.64 rad/s** |
| Stall current | 1.2 A @ 4.8V | About 7.2 A with all 6 stalled (about 9.6 A with 8 if hip roll is added); battery and wiring must handle it |
| Rated / no-load current | 500 mA / ≤ 160 mA @ 4.8V | — |
| Resolution | 0.088° (360°/4096) | The simulation must include this quantization |
| Backlash | ≤ 0.5° | — |
| Gear ratio | 1/320, metal gears | Coreless motor, ball bearings |
| Range | 360° (no limit), plus a multi-turn mode | Joint range is set by hard stops on the printed parts |
| Weight / size | 21 ± 2 g / 34 × 20 × 23 mm | 126 g for 6 (168 g for 8 if hip roll is added) |
| Horn spline | 25T, 4.95 mm OD | Use the stock horn as the mechanical interface |
| Case | PA66 + 43% glass fiber | — |
| Cable length | 15 ± 0.5 cm | Account for it in cable routing |
| Communication | Half-duplex async serial, 38400 bps–1 Mbps, IDs 0–253 | — |
| Feedback | Position, speed, load, input voltage, current, temperature | On the real robot, current can be used to estimate torque |
| Modes | Angle servo, constant-speed motor, **constant-current motor**, multi-turn | Constant-current mode amounts to torque control; see item 3 below |

All design and simulation work uses the official 4.8 V values above. A higher supply voltage gives more torque and speed, but there are no official numbers; measure on a test bench before relying on them.

### Design rules that follow

1. **Torque budget** (every joint must satisfy it):
   - Static load (gravity plus ground reaction) p99 over the full gait cycle ≤ **0.216 Nm** (rated load)
   - Measured torque p99 in simulation ≤ **0.62 Nm** (70% of stall)
   - Time spent at ≥ 95% of the limit < 2%
   - In the moonwalk, the ankle of the toe-raised foot must hold torque **continuously** (the mean torque has a fixed sign and does not cancel out). A coreless servo held near stall for long overheats, so the rated load is the real ceiling.
2. **Size**: static torque ∝ mass × leg length. Reference data: a robot of the same configuration at 15.1 kg with 0.55 m legs needs static moonwalk torques of 10 Nm at the hip, 7.6 Nm at the knee and 6.5 Nm at the ankle. Scaling from that, the suggested starting point is **0.7 kg with 0.18 m legs**:

   | Design point | Hip pitch | Hip roll (optional) | Knee | Ankle |
   | --- | --- | --- | --- | --- |
   | 0.7 kg, 0.18 m legs (suggested start) | 0.15 Nm | ~0.15 Nm | 0.12 Nm | 0.10 Nm |
   | Share of 0.216 Nm rated load | 70% | 70% | 53% | 46% |
   | (Reference) original 0.8 kg, 0.20 m | 0.19 Nm (89%) | ~0.19 Nm | 0.15 Nm | 0.13 Nm |

   This table is a **scaled estimate**. Phase 1 must recompute it from this project's own geometry and masses (see cad/CLAUDE.md).
3. **Control mode**:
   - **Angle-servo mode is the primary mode**: the controller outputs joint target angles and the servo's internal PID tracks them, targeting a 100 Hz update rate. The simulation must model the servo itself (see "Servo model" in train/CLAUDE.md).
   - **Constant-current mode is an option**: it lets the real robot command torque directly, e.g. for center-of-pressure control at the flat foot's ankle (see hard rule 13 in train/CLAUDE.md). But the torque loop has to be closed on the host over the bus, so first measure how fast 6 servos can be read and written over half-duplex 1 Mbps. The current-to-torque conversion also needs bench calibration.
   - Current feedback lets the real robot estimate each joint's torque, to compare against the torque logs from simulation.
4. **No external gear reduction**: a 10:1 reduction would multiply torque but drop speed to about 44°/s. With joint speed limited to around 30°/s, the moonwalk gait fails completely.

---

## Shared conventions

**Coordinates and pose**

- Z up, +Y forward. The origin is on the ground midway between the feet; the ground is z = 0 when standing straight
- **Zero pose = standing straight**: with all joints at 0°, the legs are fully extended and the feet are flat on the ground. The standing pose with the knees pre-bent 15–30° lives in `config/gait.yaml`
- The robot moves toward −Y while moonwalking. Every directional metric (e.g. backward distance) must state what its sign means

**DOF and joints** (3 active DOF per leg by default, all three in the sagittal plane)

| Joint | Name (left leg; right uses `R_`) | Default range | Positive direction |
| --- | --- | --- | --- |
| Hip pitch (flexion/extension) | `L_hip_pitch` | −30° to 90° | Positive = leg swings forward |
| Knee pitch | `L_knee_pitch` | 0° to 120° | 0° = fully extended; positive = human-style knee flexion (heel moves back) |
| Ankle pitch (dorsi/plantarflexion) | `L_ankle_pitch` | −35° to +30° | Negative = plantarflexion (toe down); final range set in phase 1 |
| Foot wheel (passive) | `L_wheel_axle` | Continuous | Not counted as an active DOF (to be confirmed with the advisor) |
| Hip roll (ab/adduction, **optional**) | `L_hip_roll` | −20° to 20° | Not installed by default; if added, define each side explicitly (mirroring reverses rotation about Y) |

- Link names: `base_link`, `L_thigh`, `L_shank`, `L_foot`, `L_wheel`. If hip roll is added, an `L_hip` link sits between `base_link` and `L_thigh`
- Joint axes, positive directions and ranges are all defined in `config/robot.yaml` and shared by modeling and simulation; neither side defines its own. Whether hip roll is installed is a single switch in `robot.yaml`, read by both modeling and simulation

**Why hip roll is off by default**

- The moonwalk keeps both feet on the ground the whole time. With both feet flat and no ankle roll, the two legs and the ground form a closed chain: a ±15° hip-roll command moves the pelvis only about 0.1 mm, and forcing one leg longer tips the pelvis over. Shifting the pelvis sideways with both feet down would need hip roll plus ankle roll, i.e. 5 DOF per leg, which exceeds the limit. So hip roll can do very little in this gait.
- Dropping it saves 2 servos and 42 g (about 6% of 0.7 kg), which leaves more torque margin at the other joints. The hip-pitch servo can mount directly to the pelvis, removing one link and one layer of backlash. The action space shrinks from 8 to 6 dimensions.

**Consequences of having no hip roll**

- The robot has **no active lateral correction at all**. Lateral stability must hold **statically**, from stance width and foot width alone (lateral margin ≥ 1 cm in every phase)
- **Assembly errors cannot be fixed in software**: a 1° lateral tilt lifts one edge of a 30 mm-wide foot by about 0.5 mm. This has to be handled by print accuracy, shims or a softer sole material
- Only fore-aft, both-feet-down motions like the moonwalk are possible; standing on one leg, side-stepping or ordinary walking all need hip roll

**Conditions for adding hip roll** (if any one is met, report it; the user decides whether to add it)

1. In the phase 1 analysis, no acceptable stance width and foot width reaches a lateral static margin ≥ 1 cm
2. In phase 3, falling sideways is the main failure mode, and widening the stance or the feet does not fix it
3. On the real robot, lateral foot tilt cannot be absorbed by shims or sole material and makes contact unstable
4. The user decides to extend the robot beyond the moonwalk

**What adding it involves**: turn on the switch in `robot.yaml` → rerun the phase 1 analysis (42 g heavier, so the torque budget must be recomputed) → regenerate the model → re-import and rerun acceptance. Trained policies cannot be reused; the action space changes, so retrain.

**Division of labor in the foot** (details in cad/CLAUDE.md for foot design and train/CLAUDE.md for foot simulation)

- The toe-raised foot carries the weight and must not move, so the **toe pad grips**. The flat foot has to slide backward, so the **sole (or foot wheel) slides**. This is the opposite of intuition
- The ankle angle selects what touches the ground: when the foot is flat only the sliding part touches; past about 10° of plantarflexion the toe pad takes over
- The primary design is a **one-way foot wheel** (rolls backward, locks forward), with a PTFE sole as the control. Which one is adopted is decided by both versions' acceptance results in phase 3

---

## Project structure and hand-off

```
moonwalk/
  CLAUDE.md               # This file: shared spec
  README.md
  config/
    robot.yaml            # Dimensions, materials and part masses, joints, servo parameters, friction, collision shapes, print parameters, electronics placeholders
    gait.yaml             # Gait parameters, standing pose
  cad/                    # Phases 1 and 2 (details in cad/CLAUDE.md)
    CLAUDE.md
    analysis/             # Sizing feasibility analysis
    vendor/               # Vendor files (servo STEP); do not modify
    ...                   # build123d modeling code
    out/                  # Generated: stl/, step/, views/, urdf/, model_report.md
  train/                  # Phase 3 (details in train/CLAUDE.md)
    CLAUDE.md
    assets/               # cad/out/urdf → USD (the only code not belonging to a version)
    v1/, v2/ …            # Each model version and its code are independent; never modify another version
  hardware/
    bom.md                # Bill of materials
    print_log.md          # Tolerance test results, print settings
  firmware/               # After phase 4
```

**Hand-off rules**

- **The only sources of truth are `config/`, the code in `cad/`, and the vendor files in `cad/vendor/`.** Everything else (STL, STEP, URDF, USD) is generated and never edited by hand; the vendor files in `cad/vendor/` are not modified either.
- **`cad/` produces → `train/` reads**: `train/` only reads `cad/out/urdf/` and `cad/out/model_report.md`. It does not modify them, and it does not modify the code in `cad/`.
- **Who owns which part of `robot.yaml`**:
  - Geometry, materials, part masses, collision shapes and print parameters belong to `cad/`
  - Servo parameters, friction coefficients and joint ranges are shared
  - `gait.yaml` belongs to `train/`
- **Every version in `train/` has its own snapshot**: when a version is created, `config/` and the USD are copied into `train/vN/snapshot/`, so later changes to `cad/` or `config/` do not change the behavior of existing versions
- **When simulation reveals a mechanical problem** (e.g. not enough torque, a foot that cannot reach the floor, interference), `train/` reports the measured numbers. Size changes go back through `cad/` (rerun the analysis and `cad/build.py`); do not work around them on the simulation side.
- **Separate environments**: modeling uses its own conda environment (`cad`); simulation uses Isaac Sim's bundled Python. The two sides exchange files only and never import each other.

---

## General rules

1. **If a unit is uncertain, measure it.** For example, PhysX's `maxJointVelocity` is in deg/s; setting it to 30 thinking it is rad/s locks the joints at 30°/s, keeps torque saturated, and the simulation raises no error at all. For any property whose unit is uncertain, measure its actual effect with a small experiment instead of guessing from documentation.
2. **Calibrate the sign of any directional metric against a sample with a known correct answer.** A flipped sign raises no error, and all downstream analysis looks reasonable until an outside fact contradicts it.
3. **If a change produces no change in the metric at all, look one layer down.** A complete non-response is itself evidence that the problem lies below the layer being edited.
4. **Report numbers with their measurement conditions**: harness or not, duration, number of environments, which model version, which foot type.
