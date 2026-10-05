"""cad/out/urdf -> USD with Isaac Lab's UrdfConverter (train/CLAUDE.md "Importing into Isaac Lab", step 1).

    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV \
        ~/Desktop/IsaacLab/isaaclab.sh -p ~/Desktop/moonwalk/train/assets/import_urdf.py [--variant wheel|ptfe|both]

Output: train/assets/build/<YYYYMMDD-HHMMSS>_<variant>_<total mass>g/ — a new folder every run, never
overwritten. The importer writes a layered USD (interface file + payloads/), so a version snapshot must copy
the whole folder. Settings: floating base, fixed joints merged, no collisions from visual meshes, joint drives
with stiffness = damping = 0 (the custom actuators in each version are the only source of joint torque),
PhysX physics variant selected.
"""

import argparse
import re
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

ROOT = Path(__file__).resolve().parents[2]
URDF_DIR = ROOT / "cad" / "out" / "urdf"
REPORT = ROOT / "cad" / "out" / "model_report.md"
BUILD = Path(__file__).resolve().parent / "build"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--variant", default="both", choices=["wheel", "ptfe", "both"])
add_launcher_args(parser)
args = parser.parse_args()
args.require_kit = True                       # the Isaac Sim URDF importer extension needs Kit
args.physics = "isaacsim_physx"

import isaaclab.sim as sim_utils  # noqa: E402,F401
from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def total_mass(variant: str) -> float:
    """Total mass (g) of a foot variant as printed in model_report.md ('### <variant> foot — total **X g**')."""
    m = re.search(rf"### {variant} foot — total \*\*([0-9.]+) g\*\*", REPORT.read_text())
    if not m:
        raise RuntimeError(f"total mass for '{variant}' not found in {REPORT}")
    return float(m.group(1))


def convert(variant: str) -> Path:
    urdf = URDF_DIR / f"moonwalk_mini_{variant}.urdf"
    if not urdf.exists():
        raise FileNotFoundError(urdf)
    out = BUILD / f"{time.strftime('%Y%m%d-%H%M%S')}_{variant}_{total_mass(variant):.1f}g"
    if out.exists():
        raise FileExistsError(f"{out} exists; build outputs are never overwritten")
    cfg = UrdfConverterCfg(
        asset_path=str(urdf),
        usd_dir=str(out),
        fix_base=False,
        merge_fixed_joints=True,
        collision_from_visuals=False,
        self_collision=False,
        force_usd_conversion=True,
        make_instanceable=True,
        physics_variant=UrdfConverterCfg.PhysicsVariant.PHYSX,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    conv = UrdfConverter(cfg)
    (out / "SOURCE.txt").write_text(
        f"urdf: {urdf.relative_to(ROOT)}\nusd: {Path(conv.usd_path).relative_to(out)}\n"
        f"total_mass_g: {total_mass(variant)}\nconverted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"isaaclab UrdfConverter, fix_base=False, merge_fixed_joints=True, collision_from_visuals=False, "
        f"drive target none (stiffness=damping=0), physics variant physx\n")
    print(f"[import_urdf] {variant}: {conv.usd_path}", flush=True)
    return out


def main():
    variants = ["wheel", "ptfe"] if args.variant == "both" else [args.variant]
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args):
        for v in variants:
            convert(v)


if __name__ == "__main__":
    main()
