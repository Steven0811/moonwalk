"""Headless orthographic renderer for the review images (matplotlib, painter's algorithm, flat shading)."""
from __future__ import annotations

import numpy as np

from assembly import fk, to_location
from links import link_bodies, linkage, linkage_names, links

COLORS = {
    "petg_structural": (0.62, 0.70, 0.80),
    "petg_light": (0.82, 0.84, 0.78),
    "tpu_85a": (0.20, 0.20, 0.22),
    "servo": (0.10, 0.10, 0.12),
    "elec_battery": (0.95, 0.70, 0.20),
    "elec_controller": (0.15, 0.55, 0.30),
    "elec": (0.35, 0.65, 0.45),
    "steel": (0.75, 0.75, 0.78),
    "hardware": (0.5, 0.5, 0.5),
}


def _color(b):
    if b.visual is not None or "servo" in b.name:
        return COLORS["servo"]
    if b.name.startswith("elec_"):
        return COLORS.get(b.name, COLORS["elec"])
    if b.name == "axle" or any(k in b.name for k in ("bearing", "clutch", "pin")):
        return COLORS["steel"]
    return COLORS.get(b.material, COLORS["hardware"])


def _subdivide(T, max_edge=3.0):
    """Split triangles until every edge is ≤ max_edge (keeps the painter's depth sort reliable)."""
    out = []
    while len(T):
        e = np.max(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), axis=1)
        small = e <= max_edge
        out.append(T[small])
        B = T[~small]
        if not len(B):
            break
        a, b, c = B[:, 0], B[:, 1], B[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        T = np.concatenate([np.stack(x, axis=1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def _tris(shape, tol, max_edge=3.0):
    v, t = shape.tessellate(tol, 0.3)
    V = np.array([[p.X, p.Y, p.Z] for p in v])
    T = V[np.array(t)] if len(t) else np.zeros((0, 3, 3))
    return _subdivide(T, max_edge)


def scene(variant: str, q: dict | None = None, base=None, tol=0.15, show_hidden=False):
    """[(triangles (n, 3, 3) in world mm, rgb)] for the posed robot (display models of servos)."""
    T = fk(q or {}, variant, base)
    out = []
    for name, side, kind in links(variant):
        M = T[name]
        for b in link_bodies(kind, side, variant):
            if not b.collide and b.visual is None:
                continue                                    # lumped hardware masses are not drawn
            if b.name in linkage_names():
                continue
            shp = b.visual if b.visual is not None else b.shape
            tr = _tris(shp, tol)
            tr = tr @ M[:3, :3].T + M[:3, 3]
            out.append((tr, _color(b)))
    for side in ("L", "R"):
        M = T[f"{side}_shank"]
        for b in linkage((q or {}).get(f"{side}_ankle_pitch", 0.0), side):
            tr = _tris(b.shape, tol) @ M[:3, :3].T + M[:3, 3]
            out.append((tr, _color(b)))
    return out


VIEWS = {
    #            view dir (camera looks along), up
    "front": (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "side": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),     # from the robot's right: +Y to the right
    "top": (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
    "iso": (np.array([-1.0, -1.2, -0.75]), np.array([0.0, 0.0, 1.0])),
    "iso_back": (np.array([1.0, 1.2, -0.6]), np.array([0.0, 0.0, 1.0])),
}


def draw(ax, sc, view, ground=True, light=None):
    from matplotlib.collections import PolyCollection
    d, up = VIEWS[view] if isinstance(view, str) else view
    d = d / np.linalg.norm(d)
    r = np.cross(d, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, d)
    L = -d + 0.5 * u + 0.3 * r if light is None else light
    L /= np.linalg.norm(L)
    polys, cols, depth = [], [], []
    for tr, c in sc:
        if len(tr) == 0:
            continue
        n = np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0])
        nn = np.linalg.norm(n, axis=1)
        keep = nn > 1e-12
        tr, n, nn = tr[keep], n[keep], nn[keep]
        n = n / nn[:, None]
        # tessellation winding is not consistent across faces: no culling, two-sided shading
        shade = 0.35 + 0.65 * np.abs(n @ L)
        xy = np.stack([tr @ r, tr @ u], axis=-1)
        polys.append(xy)
        cols.append(np.clip(np.array(c)[None, :] * shade[:, None] + 0.08 * (1 - shade[:, None]), 0, 1))
        depth.append((tr @ d).mean(axis=1))
    P = np.concatenate(polys)
    C = np.concatenate(cols)
    D = np.concatenate(depth)
    o = np.argsort(-D)
    pc = PolyCollection(P[o], facecolors=C[o], edgecolors=C[o], linewidths=0.15)
    ax.add_collection(pc)
    if ground and view in ("front", "side"):
        ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    return P


def save_views(variant: str, outdir, q=None, base=None, tag="", tol=0.15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    sc = scene(variant, q, base, tol)
    paths = []
    for v in ("front", "side", "top", "iso", "iso_back"):
        fig, ax = plt.subplots(figsize=(7, 7))
        draw(ax, sc, v)
        ax.set_title(f"Moonwalk Mini — {v} view ({variant} foot){tag}", fontsize=10)
        if v in ("front", "side"):
            ax.text(0.01, 0.01, "mm grid 10" if False else "", transform=ax.transAxes)
        p = outdir / f"{v}_{variant}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths
