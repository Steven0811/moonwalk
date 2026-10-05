"""Zoned physics materials for an imported USD (train/CLAUDE.md "Importing into Isaac Lab", step 2).

    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV \
        ~/Desktop/IsaacLab/isaaclab.sh -p ~/Desktop/moonwalk/train/assets/postprocess_usd.py [build folder ...]

With no argument it processes the newest wheel and ptfe folders in train/assets/build/. A folder is processed
once (marker POSTPROCESSED.txt); a second run refuses rather than overwrite.

Writes only what the Isaac Lab configs cannot set: one physics material per zone, bound (purpose "physics")
to every collision primitive whose name matches robot.yaml `collision.names`, with friction and restitution
combine mode "min". Coefficients come from robot.yaml `friction:` (estimates until measured); they are never
tuned here.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import yaml
from pxr import Sdf, Usd, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(__file__).resolve().parent / "build"


def zone_of(name: str, names: dict) -> str:
    """Zone of a collision prim from robot.yaml collision.names (templates with {side}, {lr}, {link})."""
    for zone, tmpl in names.items():
        pat = "^" + re.escape(tmpl).replace(r"\{side\}", "(L|R)").replace(r"\{lr\}", "(in|out)") \
            .replace(r"\{link\}", r"\w+") + "$"
        if re.match(pat, name):
            return zone
    if name.startswith("base_link_"):
        return "body"
    raise KeyError(f"collision prim '{name}' matches no robot.yaml collision.names template")


def materials_for(variant: str, fr: dict) -> dict:
    """zone -> (material name, friction). The ptfe heel is part of the PTFE sole."""
    return {
        "toe_pad": ("tpu_toe_pad", fr["toe_pad"]),
        "wheel": ("tpu_tyre", fr["wheel_tyre"]),
        "sole": ("ptfe_sole", fr["ptfe_sole"]),
        "heel": ("ptfe_sole", fr["ptfe_sole"]) if variant == "ptfe" else ("petg_structure", fr["structure"]),
        "body": ("petg_structure", fr["structure"]),
    }


def process(folder: Path, robot: dict) -> None:
    marker = folder / "POSTPROCESSED.txt"
    if marker.exists():
        raise FileExistsError(f"{folder} already post-processed; re-import to get a fresh folder")
    variant = "wheel" if "_wheel_" in folder.name else "ptfe"
    usd = next(folder.glob("*/*.usda"))
    stage = Usd.Stage.Open(str(usd))
    root = stage.GetDefaultPrim().GetPath()
    mats = materials_for(variant, robot["friction"])
    made = {}
    for mname, mu in set(mats.values()):
        path = root.AppendPath(f"PhysicsMaterials/{mname}")
        mat = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(float(mu))
        api.CreateDynamicFrictionAttr(float(mu))
        api.CreateRestitutionAttr(0.0)
        p = mat.GetPrim()
        p.AddAppliedSchema("PhysxMaterialAPI")
        p.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
        p.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
        made[mname] = mat
    rows = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        zone = zone_of(prim.GetName(), robot["collision"]["names"])
        mname, mu = mats[zone]
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(made[mname], UsdShade.Tokens.weakerThanDescendants, "physics")
        rows.append(f"{prim.GetName():22s} {prim.GetTypeName():7s} zone={zone:8s} material={mname} mu={mu}")
    stage.GetRootLayer().Save()
    marker.write_text(f"post-processed {time.strftime('%Y-%m-%d %H:%M:%S')}, variant {variant}\n"
                      "friction/restitution combine mode: min; restitution 0\n" + "\n".join(rows) + "\n")
    print(f"[postprocess_usd] {folder.name}: {len(rows)} collision prims bound", flush=True)
    print("\n".join("    " + r for r in rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*")
    a, _ = ap.parse_known_args()
    with open(ROOT / "config" / "robot.yaml") as f:
        robot = yaml.safe_load(f)
    if a.folders:
        folders = [Path(p) for p in a.folders]
    else:
        folders = [sorted(BUILD.glob(f"*_{v}_*g"))[-1] for v in ("wheel", "ptfe")]
    for fo in folders:
        process(fo, robot)


if __name__ == "__main__":
    sys.exit(main())
