"""Common start-up for v1 tools: argument parsing with the Isaac Lab launcher, PhysX backend, v1 on sys.path."""
import argparse
import sys
from pathlib import Path

VDIR = Path(__file__).resolve().parents[1]
if str(VDIR) not in sys.path:
    sys.path.insert(0, str(VDIR))           # only this version's folder (version rule 3)


def parser(desc: str) -> argparse.ArgumentParser:
    from isaaclab.app import add_launcher_args
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--variant", default="wheel", choices=["wheel", "ptfe"])
    add_launcher_args(p)
    return p


def finish_args(p):
    a = p.parse_args()
    a.physics = "isaacsim_physx"            # explicit backend (train/CLAUDE.md runtime rules)
    return a
