"""Show the robot in VS Code's OCP CAD Viewer.

1. Install the VS Code extension "OCP CAD Viewer" and open its viewer (OCP icon in the side bar).
2. Select the `cad` conda env as the Python interpreter, then run this file (Run Python File), or:
       conda run -n cad python cad/show.py --variant wheel --pose stand
   Poses: zero | stand | slide_start | slide_end | transfer | raise | limits

Every link is shown as its own group (servos with the vendor model), so parts can be hidden in the tree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))

from build123d import Compound  # noqa: E402

from assembly import fk, to_location  # noqa: E402
from links import link_bodies, linkage, linkage_names, links  # noqa: E402
from params import P  # noqa: E402

COL = {"petg_structural": "#9fb3cc", "petg_light": "#d6d8c7", "tpu_85a": "#333333", "purchased": "#222222"}


STEEL = ("bearing", "clutch", "pin", "axle")


def _col(b):
    if any(k in b.name for k in STEEL):
        return "#c0c4cc"
    if b.name.startswith("elec_"):
        return "#e8a33a" if "battery" in b.name else "#2e8b57"
    return COL.get(b.material, "#888888")


def pose_q(variant: str, pose: str):
    if pose == "zero":
        return {}, None
    if pose == "limits":
        return {"L_hip_pitch": 60, "L_knee_pitch": 110, "L_ankle_pitch": -35,
                "R_hip_pitch": -30, "R_knee_pitch": 0, "R_ankle_pitch": 30}, None
    from gait import GaitModel
    from model import load
    r, g = load()
    gm = GaitModel(r, g, variant)
    if pose == "stand":
        from kinematics import foot_pose
        st = g["standing"]
        an, p = foot_pose(gm.fg, 0.0, gm.theta[st["foot_state"][variant]])
        py, H = st["pelvis_y"][variant], st["hip_height"][variant]
        qh, qk = gm.leg.ik(an[0] - py, an[1] - H)
        qa = p - (qh - qk)
        q = {f"{s}_{k}": np.degrees(v) for s in ("L", "R") for k, v in
             (("hip_pitch", qh), ("knee_pitch", qk), ("ankle_pitch", qa))}
        base = np.eye(4)
        base[:3, 3] = (0, py, H)
        return q, base
    k = [k for k in gm.kf if k["name"] == pose][0]
    s = gm.pose(k["t"], gm.default_pelvis(), 0)
    q = {}
    for lg in s.legs:
        sd = "L" if lg.x < 0 else "R"
        q.update({f"{sd}_hip_pitch": np.degrees(lg.qh), f"{sd}_knee_pitch": np.degrees(lg.qk),
                  f"{sd}_ankle_pitch": np.degrees(lg.qa)})
    base = np.eye(4)
    base[:3, 3] = (0, s.pelvis[0], s.pelvis[1])
    return q, base


def _find_viewer():
    """Port of a running OCP CAD Viewer: registered ones in ~/.ocpvscode first, then the default 3939."""
    import json
    import socket
    ports = []
    reg = Path.home() / ".ocpvscode"
    if reg.exists():
        try:
            ports = [int(p) for p in json.loads(reg.read_text()).get("services", {})]
        except Exception:  # noqa: BLE001
            ports = []
    for p in ports + [3939]:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", p)) == 0:
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=P.foot_variant)
    ap.add_argument("--pose", default="stand")
    ap.add_argument("--simplified", action="store_true", help="show simplified servos instead of the vendor model")
    ap.add_argument("--port", type=int, default=None, help="viewer port (default: auto-detect, then 3939)")
    a, _ = ap.parse_known_args()
    port = a.port or _find_viewer()
    if port is None:
        print("No OCP CAD Viewer is running.\n"
              "In VS Code: click the OCP CAD Viewer icon in the left activity bar (or Ctrl+Shift+P →\n"
              "'OCP CAD Viewer: Open viewer'), wait for the 'Viewer' tab to appear, then run this again.\n"
              "If the viewer shows a port other than 3939, pass it with --port <port>.")
        raise SystemExit(1)
    from ocp_vscode import set_port, show
    set_port(port)
    q, base = pose_q(a.variant, a.pose)
    T = fk(q, a.variant, base)
    objs, names, colors = [], [], []
    for name, side, kind in links(a.variant):
        loc = to_location(T[name])
        for b in link_bodies(kind, side, a.variant):
            if not b.collide and b.visual is None:
                continue
            if b.name in linkage_names():
                continue
            shp = b.visual if (b.visual is not None and not a.simplified) else b.shape
            objs.append(shp.moved(loc))
            names.append(f"{name}/{b.name}")
            colors.append(_col(b))
    for side in ("L", "R"):
        loc = to_location(T[f"{side}_shank"])
        for b in linkage(q.get(f"{side}_ankle_pitch", 0.0), side):
            objs.append(b.shape.moved(loc))
            names.append(f"{side}_linkage/{b.name}")
            colors.append(_col(b))
    show(*objs, names=names, colors=colors)


if __name__ == "__main__":
    main()
