# train/v1 — rule-based moonwalk controller (wheel foot first)

- **Parent:** none (first version).
- **Purpose:** phase 3 steps 2–8 of `train/CLAUDE.md`: Isaac Lab environment (PhysX, 1 kHz physics, 100 Hz
  control), custom servo actuator, one-way foot-wheel model, standing test, the four-phase rule-based gait (slow
  4 s cycle first), torque logging and acceptance.
- **Priority:** the **wheel** foot first (user request, 2026-10-04). The PTFE foot is in the same snapshot and gets
  the same runs for the comparison the spec requires.
- **No reinforcement learning in this version.** If RL is needed it goes in a new version (hard rule 7).

## Snapshot

`snapshot/robot.yaml`, `snapshot/gait.yaml` (config at commit c6abd0d) and `snapshot/usd/` (both foot variants,
imported 2026-10-04 with `train/assets/import_urdf.py` + `postprocess_usd.py`; `check_model.py` PASS for both, see
`CHECK_MODEL_*.txt` in each USD folder). This version reads only its snapshot.

## Runtime

Isaac Lab 3.0.0 on Isaac Sim 6.1.0-rc.26, backend **Isaac Sim PhysX**, dt 1 ms, control 100 Hz (decimation 10).
No extra packages installed into Isaac Sim's Python. `SimulationCfg.use_newton_actuators=False` is set explicitly:
Isaac Lab 3.0 defaults it to True even on PhysX, which hands explicit actuators to a native path instead of the
Python servo model (the servo step test confirms compute() runs once per physics step: 600 calls / 600 steps).

## How to run (conda deactivated)

```
I=~/Desktop/IsaacLab/isaaclab.sh; E="env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV -u PYTHONPATH"
$E $I -p tools/test_servo_step.py            # step 2
$E $I -p tools/test_wheel_incline.py         # step 3
$E $I -p tools/stand_test.py --variant wheel # step 4
$E $I -p tools/run_gait.py --variant wheel --seconds 60 [--cycle 2.0] [--video]   # steps 5-7
~/Desktop/IsaacLab/_isaac_sim/python.sh tools/validate.py runs/gait/<run>         # acceptance table
~/Desktop/IsaacLab/_isaac_sim/python.sh tools/measure_torque.py runs/gait/<run>   # torque report
~/Desktop/IsaacLab/_isaac_sim/python.sh tools/cycle_profile.py runs/gait/<run>    # margin / angles / torque folded into one cycle (CSV)
$E $I -p tools/robustness.py --variant wheel --envs 64 [--no-random]               # robustness / rule-3 check
```

## Model decisions and findings (with the measurement that forced each)

1. **Wheel clutch = ratchet on the unwrapped axle angle, applied through the implicit PhysX drive.** The wheel set
   inertia is 1.7e-8 kg·m²; an explicit torque at 1 kHz is only stable below ≈ 3.4e-5 N·m·s/rad of damping, far too
   weak to lock the wheel. A speed-triggered lock leaked forward through small back-and-forth motions; the ratchet
   (implicit spring to the most-backward angle reached) does not.
2. **The wheel angle wraps at ±π** (continuous joint). The ratchet tracks an unwrapped angle; before this fix the
   wrap looked like a 6 rad forward jump and the lock spring threw the wheels around.
3. **Wheel armature 2e-6 kg·m².** With the bare inertia the lock was under-resolved by the articulation solver
   under contact load (clutch slip 4.8 mm at the tyre on a 10° incline). With armature: 0.15 mm. 2e-6 is ~15 % of
   the robot's own reflected rolling inertia.
4. **Clutch direction sign = -1** (forward rolling = negative joint speed), calibrated with the incline test. An
   earlier run that seemed to say +1 was invalid: the test robot was tipping over and being auto-reset.
5. **Rolling resistance** is the robot.yaml rolling coefficient + axle Coulomb + axle damping, applied as an
   equivalent viscous damping at the 40 mm/s design slide speed (explicit Coulomb chatters on this inertia). The
   incline test's terminal speed matches the prediction from it within 1 %.
6. **Servo backlash** is modelled as a dead band on the position error (no restoring torque inside the play);
   ankle = servo 0.5° + linkage 0.2°.
7. **Ankle target (hard rule 12): measured hip and knee, pelvis assumed level ("hipknee").** Adding the measured
   pelvis pitch made the robot fall at 1.4 s (positive feedback through the ground: with both feet down a pelvis
   tilt plantarflexes both ankles, which tilts the pelvis further — the situation of hard rule 13). Planned angles
   and hipknee both walked 20 s; hipknee follows the rule's wording.
8. **Start-up:** standing pose → the cycle's `slide_mid` keyframe in 1.5 s after a 0.5 s hold. A first start-up
   that slid B backward with A in the switch posture tipped forward (once B is flat it only touches on its axle
   line). Not in gait.yaml; built from its quantities.
9. **Keyframe interpolation = smoothstep** (zero velocity at keyframes). Linear and smoothstep gave the same
   margins; smoothstep kept for smoother trajectories.

## Results (2026-10-04)

All runs: Isaac Sim PhysX, dt 1 ms, control 100 Hz, no harness, no viewer, servo model (latency 5–10 ms random per
reset unless stated, backlash per robot.yaml, 0.088° quantisation), zoned friction from robot.yaml, rule-based v1
(ankle mode hipknee, smoothstep keyframes, start-up standing → slide_mid). Run folders under `runs/` (not in git).

| Step | Test | Wheel | PTFE |
|---|---|---|---|
| 2 | Servo step (fixed base, zero g) | PASS: 600 calls / 600 steps, peak 7.640 rad/s, peak 0.8800 N·m, latency 7 steps for 7 ms | — |
| 3 | Incline (staggered ±30 mm, H 140 mm) | PASS: back 2° rolls 72.9 mm, terminal 18.5 vs 18.4 mm/s predicted; fwd 2° / 10° clutch slip 0.036 / 0.152 mm | — |
| 4 | Stand 30 s | PASS: max err 0.46°, max τ 0.021 N·m, no saturation | PASS: max err 0.61°, max τ 0.027 N·m |
| 5 | Gait 60 s, 4 s cycle, zoned materials | PASS: no fall, 645–700 mm backward | no fall, but see step 8 |
| 6 | Loaded flat foot at weight transfer | **rolls back up to 13.7 mm (mean worst 8.5 mm) per transfer** on the free-backward wheels | sole slides; support slips forward up to 9.9 mm |
| 7 | Speed-up, open loop | 3 s cycle PASS (60 s, slides 40.4 mm); 2 s falls at 7.9 s; 1.5/1.2/1.0 s fall in 1–4 s | not run |
| 8 | Acceptance (60 s, 4 s cycle) | see below | slides 10 mm (target 40), L/R 2.4/17.6 mm → FAIL |
| 8 | Robustness, 64 envs randomized | **PASS: 64/64 stand 60 s**, 622–746 mm backward | not run |
| 3 (rule) | Vectorized vs single env | 4 nominal envs 680–706 mm vs single 700 mm, all stand → consistent | — |

Wheel acceptance (60 s, 4 s cycle, 1 env, smoothstep run): stand 60 s ✔; slide 40.7 mm (target 40) ✔; L/R slide
2.1 % ✔; support slip during the slide 0.99 mm ✔; foot-zone wrong contact 0.0 % ✔; tracking max 2.89° ✔; torque RMS
≤ 0.041, p99 ≤ 0.117 N·m, no saturation ✔; ankle of the toe-raised foot holds −0.051 N·m (24 % of rated);
speed p99 94°/s ✔; **static margin from the real contacts: min −2.1 / −2.4 mm, p5 1.7 / 3.0 mm vs phase-1 14.1 mm
✘** (the polygon collapses to a triangle for ~0.1 s after each role swap: the robot rocks onto foot edges);
robustness 64/64 ✔. Not yet measured: standing for ≥ 95 % of *starting phases* (only the standing-pose start was
run), the visual heel-toe comparison (needs video, see open items).

PTFE finding: the phase-1 pelvis keyframes maximise the static margin, which for the PTFE foot puts the CoM over
the slider's long flat sole: during the slide the slider carries 82 % of the weight (3.06 of 3.75 N), its PTFE
friction is then comparable to what the lightly loaded toe pad can hold, and both feet slip. Fix = PTFE keyframes
with a load-share constraint (support toe carries most of the weight while sliding); that is a gait.yaml change,
so it belongs in a new version.

## Open items

- Video recording needs MoviePy in Isaac Sim's Python (not installed; waiting for the user's OK).
- Step 7 balance feedback (IMU / ankle centre-of-pressure control, hard rule 13 a vs b) for cycles < 3 s.
- Static margin shortfall after role swaps (wheel).
- Starting-phase coverage of the standing criterion.
