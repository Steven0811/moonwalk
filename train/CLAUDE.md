# train/ — Isaac Sim Simulation, Gait and Training

This file covers phase 3 (import into Isaac Sim, gait validation, acceptance tests, and reinforcement learning if needed) and the calibration and robustness parts of phase 4. Motor specs, joint conventions and hand-off rules are in the root [CLAUDE.md](../CLAUDE.md).

**This folder's inputs**: `cad/out/urdf/` (URDF and meshes) and `cad/out/model_report.md`. Read only; if the model has a problem, report it so `cad/` can fix and regenerate it.

**Prerequisite**: `cad/` phase 2 has passed (`cad/checks.py` passes in full and the user has approved the shape).

---

## Runtime environment

- Isaac Sim 6.0 standalone, at `~/Desktop/isaac-sim-standalone-6.0.0-linux-x86_64`
- Run with Isaac Sim's bundled Python: `cd <isaac-sim> && env -u PYTHONPATH ./python.sh ~/Desktop/moonwalk/train/v<N>/tools/<script>.py` (the model import tools are `train/assets/<script>.py`)
- Never import code from `cad/`; read the values you need from files in `config/` or `cad/out/`
- Physics rate ≥ 1 kHz; control rate 100 Hz (matching the real robot's target bus rate)

---

## Version management

**Every model version (controller or policy) and all of its code live in their own folder — `train/v1/`, `train/v2/`, … — fully independent of each other.** For example, v1's `play_policy.py` and v2's `play_policy.py` are two different files.

### Rules

1. **Only modify the current version's folder.** While developing or training `vN`, only add or modify files under `train/vN/`. Do not touch any file in another version (including tools like `play_policy.py` and `evaluate.py`), even if it has a bug.
2. **Do not fix old versions' bugs in place.** Fix them in the current version and record in the current version's `README.md`: which version, which file, what the problem is, and whether it affects that version's known results. Only modify an old version when the user explicitly asks, and then add an "After-the-fact changes" section to that version's `README.md` stating the date, what changed and why.
3. **Versions never import each other.** A version's code may only import modules from its own folder, Isaac Sim and third-party packages. No importing another `vN`, and no `sys.path` entries pointing at other versions; there is no shared `lib/` under `train/`.
4. **Every version has its own config and model snapshot.** When a version is created, copy the current `config/robot.yaml`, `config/gait.yaml` and the USD produced by `train/assets/` into `vN/snapshot/`. The version reads only its own snapshot, so later changes to `cad/` or `config/` do not change its behavior. To use a new model or config, create a new version.
5. **All paths are relative to the version folder.** Checkpoints, logs and videos go in `vN/runs/`; no hard-coded absolute paths into other versions.

### Creating a new version

1. Copy the previous version's whole folder (except `runs/`) to the new version: e.g. `v2/` is copied from `v1/`
2. Update `snapshot/` if needed (only in the new version)
3. In the new version's `README.md`, state the parent version, what this version changes, and why
4. Fill in status and provenance in `VERSION.yaml` (see below)

When to create a new version: once the previous version is frozen, any change that alters behavior (code, reward, hyperparameters, config, model snapshot) requires a new version. Tell the user before creating one.

### Version status

Each version's `VERSION.yaml` records:

| Field | Contents |
| --- | --- |
| `status` | `open` (in development, modifiable) or `frozen` (frozen, not modifiable) |
| `parent` | Which version it was copied from (`null` for v1) |
| `created`, `frozen` | Creation date, freeze date |
| `git_commit` | The commit at freeze time |
| `snapshot` | Snapshot provenance: the date and total mass from `cad/out/model_report.md`, and the git commit of `robot.yaml` |
| `results` | The main acceptance numbers at freeze time, with their measurement conditions (number of environments, duration, harness or not, which foot type) |

- **Only one version is `open` at a time.**
- When a version has finished its final training and acceptance, the user confirms the freeze: set `status` to `frozen`, fill in the results, and commit. After that, the version's code is never modified.
- Checkpoints and videos in `runs/` are large and stay out of git; but each version's results summary in `README.md` goes into git.

### Comparing versions

- Evaluate each version with its own tools; never run an old version's checkpoint with a newer version's `evaluate.py`.
- Comparison tables between versions must state the measurement conditions of every number; numbers measured under different conditions are not compared directly.

---

## Importing into Isaac Sim

The import tools live in `train/assets/` and are the only code in `train/` that belongs to no version. They only produce USD from `cad/out/urdf/` and contain no controller or training code. Generated USD goes in `train/assets/build/`, one new file per run, named by date and the total mass from `model_report.md`, never overwriting old files. Each version copies the USD it needs into `vN/snapshot/`, so changing the import tools does not affect existing versions.

1. `import_urdf.py`: convert `cad/out/urdf/` to USD with Isaac Sim's URDF Importer. Settings: base not fixed, merge fixed joints, do not generate collisions from visual meshes.
2. `postprocess_usd.py`: write from `robot.yaml`:
   - Drives: force mode, stiffness = damping = 0, `maxForce` = stall torque 0.88 Nm
   - `maxJointVelocity` = **438 (the unit is deg/s)**
   - Armature
   - Physics materials bound by collision name (e.g. `L_foot_toe_pad`), with friction combine mode set to `min`
   - A contact sensor on each foot (normal and tangential force)
3. `check_model.py`: run after every import; if any check fails, do not proceed to simulation:
   - Total mass and per-link masses match `cad/out/model_report.md` (confirms the import changed nothing)
   - Each joint's positive direction: command +10° and measure the direction the distal end moves (positive knee = heel moves back)
   - Joint limits equal `robot.yaml`
   - All collision bodies are primitives, with no convex hulls
   - A single joint driven at full speed actually reaches about 7.64 rad/s (confirms the `maxJointVelocity` unit is right)
   - With all joints locked, the robot stands on the ground for 5 seconds without exploding, sinking through the floor or bouncing

---

## Servo model

The real robot uses angle servos, so **the actuators in simulation must behave like servos, not ideal torque motors**.

Implement it in the Python control loop, computing at every physics step:

```
tau_cmd   = Kp_servo * (theta_target - theta) - Kd_servo * theta_dot
tau_avail = tau_stall * (1 - |theta_dot| / omega_noload)   # linear torque-speed curve, applied only when torque and motion share a direction
tau       = clip(tau_cmd, -tau_avail, +tau_avail)
```

- `tau_stall` = 0.88 Nm, `omega_noload` = 7.64 rad/s (official 4.8 V values)
- Estimate `Kp_servo` and `Kd_servo` for now; identify them from bench step responses once the hardware arrives
- The simulation must include:
  - Position command latency: assume 5–10 ms for now, update after measuring
  - Gear backlash: officially ≤ 0.5°; if `cad/` uses a linkage drive, add the linkage's backlash
  - Position quantization: 0.088°
- **Constant-current mode (optional)**: if phase 3 decides that some joints (e.g. the flat foot's ankle) use constant-current mode, that joint's simulation becomes "host computes torque → clip to ±0.88 Nm → apply the torque-speed curve", updated at the measured bus rate, not the physics rate.
- **Log torque at every physics step**, not one sample per control step, and log joint speed alongside it, so you can tell whether a joint is overloaded or limited by its speed cap.

---

## Friction and feet

- **Set PhysX's friction combine mode to `min`**: the default averages, so a 0.08 sole on a 0.9 floor becomes 0.49 and does not slide at all.
- **Do not change friction coefficients at runtime through the API**, since the real robot cannot do that. "Phase-dependent friction" is achieved mechanically by the one-way foot wheel.
- Always use the measured friction coefficients in `robot.yaml` (or the estimates before measurement); never tune them to make the gait succeed.

### Simulating the foot wheel

- Each wheel is a passive revolute joint (no drive) on the foot, with its axis lateral
- The collision body is a sphere (generated by `cad/`)
- **PhysX has no rolling resistance by default**, so an untreated wheel is frictionless and more optimistic than reality. Add joint damping plus a constant Coulomb friction torque on the axle to represent bearing friction, rolling resistance and the O-ring. Estimate the values for now; measure them on the real robot with an incline rolling test and write them back
- **PhysX has no native one-way clutch**. Check the wheel's angular velocity at every physics step in the control loop: when it turns forward (the foot moving forward relative to the ground), apply a braking torque large enough to stop it; when it turns backward, apply none. This can be implemented as asymmetric damping, large in the forward direction and small in the backward direction
- Calibrate the one-way direction's sign against a known case first: put a single foot on an incline and confirm it rolls backward and locks forward

**Unresolved risk**: at the moment weight transfers from the toe-raised foot to the flat foot, the flat foot is loaded. The one-way clutch blocks forward motion but not backward; if the body's center of mass pushes that foot backward at that moment, it will roll away. This is the first thing to verify in phase 3.

**Control**: the PTFE sole variant. Run the full acceptance suite on both foot types and decide which to adopt from the measured results; do not assume the wheels are better.

---

## Gait: four-phase state machine

Phase variable φ ∈ [0,1), with a 0.5 phase offset between the legs:

| Phase | Flat foot (sliding) | Toe-raised foot (weight-bearing) | Action | Condition to advance |
| --- | --- | --- | --- | --- |
| 1. Starting stance | Dorsiflexed, flat | Plantarflexed, toe pad down | Both feet in position | Slide command issued |
| 2. Low-friction slide back | Flat; sole or wheels slide on the floor | Plantarflexed, bearing weight | Flat foot slides back; hip and knee keep body height constant | Slide distance reaches the target step length |
| 3. Weight transfer | Contact moves to the high-friction part | Goes from plantarflexed to dorsiflexed | Weight shifts from the toe-raised foot to the slid-back foot | Weight transfer complete; that ankle is back to flat |
| 4. Role swap | Becomes plantarflexed, toe-raised | Becomes dorsiflexed, flat | The two feet swap roles | Back to phase 1 with the other leg |

- Each joint's target angle is generated as a piecewise function or spline of φ, with parameters in `config/gait.yaml`
- **The slow version must be statically stable**: with a 4-second cycle, the center-of-mass projection stays inside the support polygon in every phase, with ≥ 1 cm margin both fore-aft and laterally. `cad/` already proved this offline in phase 1; in simulation, re-measure it from the actual contact points and compare with the offline report
- Dynamic balance (IMU feedback, ankle center-of-pressure control) is allowed only once the gait is sped up
- **Reset pose**: start from the standing pose in `gait.yaml` at the correct height, never from the midpoint of the joint ranges

---

## Hard rules

The "General rules" in the root file (units, signs, look one layer down on no response, state measurement conditions) also apply.

**Measurement and validation**

1. **Judge behavior only headless or from recordings.** `_APP.update()` (servicing the GUI window) advances physics on its own: the same policy that stands for the full 10 seconds headless may fall within 3 seconds in the live window. `world.render()` and recording are not affected.
2. **With a wrong reset pose, nothing responds.** If the reset pose is the midpoint of the joint ranges (a deep squat), every episode starts by falling, and any control fix leaves survival time exactly unchanged.
3. **Vectorized-environment bugs raise no errors; they return wrong numbers in a plausible range.** Examples: `get_net_contact_forces()` without `dt=` returns 0 N; foot pitch computed in the pelvis frame instead of the world frame; the reward reading the previous step's stale state. A vectorized version must match the single-environment numbers, not merely "run".
4. **Normalize accumulating metrics; lengthen the horizon for saturating ones.** Lateral drift accumulates with survival time, so divide by time or distance before comparing. When the evaluation window is too short (e.g. 4 seconds), survival time is capped at the window length and looks like the model's ceiling. Compare models at equal amounts of training.
5. **Evaluate long enough.** The gait takes several cycles to reach steady state; short runs give inaccurate slide distance and left-right symmetry. All acceptance runs are ≥ 60 seconds.
6. **Make sure the floor is big enough.** Long, many-environment evaluations can walk robots off the floor, which then scores as "sinking". Before running, check that `environments × spacing + duration × speed` fits.

**Control and learning**

7. **First get the rule-based controller through a full, slow, statically stable cycle**; do not start with reinforcement learning. If the rule-based controller cannot pass, look for the structural cause first (e.g. center of pressure always in front of the center of mass) instead of masking it with reinforcement learning.
8. **The harness is for debugging only**, and every number must state whether a harness was used. The harness supplies torques the robot cannot produce itself, so success with a harness does not mean the robot can stand on its own.
9. **Behavior cloning can teach the look of the gait, not balance**, especially when the demonstrator itself falls.
10. **Every reward term must have slope.** A Gaussian's width must span the range the policy actually visits. Before training, print each term's per-step mean; any term that has the same value across conditions that should differ (e.g. exactly a constant like 2·e⁻¹) is broken.
11. **When PPO starts from BC weights, warm up the critic too**, or the reward drops by half over the first few dozen iterations.

**Gait control**

12. **The foot's world pitch = pelvis + hip − knee + ankle**, so hip and knee errors pass straight through to the foot. Recompute the ankle target in real time from the **measured** hip and knee angles, not the planned ones.
13. **When the foot is flat, its pitch is already fixed by the ground**, and position control only fights the constraint. What the flat foot's ankle should control is where the center of pressure lands. There are two approaches, to be compared and decided in phase 3: (a) in angle-servo mode, lower Kp during this period, or adjust the center of pressure with small angle offsets; (b) switch to constant-current mode and command torque directly (limited by the bus rate; see "Servo model").
14. **Knee direction**: with a wrong joint axis, the knee bends forward (a bird's leg). `check_model.py` verifies that a positive angle is human-style knee flexion; if it fails, report it to `cad/` instead of flipping the sign on the simulation side.

---

## Acceptance criteria

| Metric | Pass criterion |
| --- | --- |
| Standing time (no harness, headless) | 60 seconds without falling, for at least 95% of starting phases |
| Distance per slide | > 90% of the target step length (60-second steady-state value) |
| Left-right symmetry | Left/right cycle time difference < 10%, slide distance difference < 10% |
| Static stability margin (slow version) | ≥ 1 cm in every phase, both fore-aft and lateral, compared against `cad/out/sizing_report.md` |
| Joint tracking error | < 5° |
| Torque | Per joint: RMS ≤ 0.216 Nm, p99 ≤ 0.62 Nm, time saturated < 2% |
| Joint speed | p99 < 70% × 438°/s (about 307°/s) |
| Support foot slip (per slide) | < 10% of the target step length; for the wheel variant, report forward and backward separately |
| Foot zone switching | Wheels actually lifted when toe-raised, toe pad actually lifted when flat; wrong-contact time from contact-point statistics < 5% |
| Robustness | With friction ±30%, mass ±15%, electronics mass ±50% and position ±1 cm, servo latency 0–15 ms, backlash 0–1°, and wheel rolling resistance ±50% randomized, standing rate still ≥ 90% |
| Visual heel-toe illusion | Side-view footage compared frame by frame against key poses of a real moonwalk |

---

## File structure

```
train/
  CLAUDE.md
  assets/                 # The only code not belonging to a version: cad/out/urdf → USD
    import_urdf.py
    postprocess_usd.py    # Writes drives, materials and sensors from robot.yaml
    check_model.py        # Post-import model checks; failure blocks simulation
    build/                # Generated USD, one new file per run, never overwritten
  v1/
    README.md             # Parent version, what changed, why, results summary, after-the-fact changes
    VERSION.yaml          # Status, provenance, freeze info, results
    snapshot/             # robot.yaml, gait.yaml and USD copied when the version was created
    scenes/               # Ground and test scenes
    lib/
      servo_model.py      # Servo model (torque-speed, latency, backlash, quantization, constant-current option)
      gait_fsm.py         # Four-phase state machine
      contact.py          # Contact points, center of pressure, support polygon
      foot_wheel.py       # Foot wheel: rolling resistance, one-way clutch
      env.py              # Simulation environment (only for reinforcement learning versions)
    tools/
      run_gait.py         # Run the gait, headless or recorded
      measure_torque.py   # Logs torque, speed and saturation at every physics step
      validate.py         # Acceptance table
      train.py            # Training (only for reinforcement learning versions)
      evaluate.py
      play_policy.py
    runs/                 # Checkpoints, logs, videos (large files stay out of git)
  v2/                     # Copied from v1; same structure, independent
  ...
```

---

## Task breakdown

**Phase 3: Import and simulation**
1. Run `import_urdf.py`, `postprocess_usd.py` and `check_model.py` in `train/assets/` to import the URDF; continue only when every check passes. Then create `train/v1/` (with `snapshot/`, `README.md` and `VERSION.yaml`); all following steps happen in the current version's folder
2. `servo_model.py`, then a single-joint step-response test confirming a maximum speed of about 7.64 rad/s and a torque limit of 0.88 Nm
3. `foot_wheel.py`, then a single-foot incline test confirming that it rolls backward, locks forward, and has the configured rolling resistance
4. Standing test: reset from the standing pose in `gait.yaml` at the correct height and stand for 30 seconds; tracking error < 2°, no saturation. Both foot types must pass
5. Four-phase state machine, slow version with a 4-second cycle, first with a single friction coefficient, then with zoned materials; no harness. Pass criterion: 60 seconds without falling
6. Run each foot type, checking first whether the loaded flat foot rolls backward at the moment of weight transfer
7. Shorten the cycle step by step to the target speed, adding IMU feedback and ankle center-of-pressure control if needed (comparing the two approaches in hard rule 13)
8. `measure_torque.py` and `validate.py`: run the full acceptance suite on both foot types, record numbers together with measurement methods, and hand the comparison to the user to choose the final foot design
   - `validate.py` must tally the direction of every fall (forward, backward, left, right). If falling sideways is the main failure mode, report it against "Conditions for adding hip roll" in the root file
9. (Optional) Consider reinforcement learning only if the rule-based controller from step 7 cannot pass acceptance, and follow hard rules 7–11. Reinforcement learning goes in a new version, not into the rule-based controller's version

**Phase 4 (calibration and robustness)**
10. Bench identification of the servos: Kp, latency, backlash; plus the current-to-torque conversion if constant-current mode is used. Write the results back to `robot.yaml`
11. After `cad/` writes back the weighed masses and regenerates the URDF, re-import and rerun `check_model.py`; create a new version with the new USD, then run acceptance
12. Add domain randomization and run the robustness acceptance test. While electronics models are not chosen, the electronics randomization (mass ±50%, position ±1 cm) must be in place from the start of phase 3, not deferred to phase 4; once models are chosen, replace the randomization center values with the weighed values
