"""Tolerance test piece: print it, try the real parts, report which size fits; the values go to robot.yaml
print: (servo_pocket_clearance, horn_recess_clearance, bearing_press, insert holes, screw holes, shaft_clear).

Layout (top view, left to right in each row; the number of notches on the edge next to a row = row number):
  row 1  M2 heat-set insert holes        Ø 3.0 / 3.2 / 3.4      (robot.yaml insert_m2_hole)
  row 2  M2.5 heat-set insert holes      Ø 3.4 / 3.6 / 3.8      (insert_m25_hole)
  row 3  M2 clearance                    Ø 2.1 / 2.3 / 2.5      (screw_m2_clear)
  row 4  M2 self-tap into print          Ø 1.5 / 1.7 / 1.9      (screw_m2_tap)
  row 5  3 mm pin (ankle / axle)         Ø 3.10 / 3.25 / 3.40  -> shaft_clear = (d - 3) / 2
  row 6  MR63 / MR52 / HF0306 bores      Ø D-0.10 / D-0.05 / D   (bearing_press)
  plus three horn recesses Ø 16.0 / 16.3 / 16.6 x 1 mm (horn_recess_clearance) and
  three servo-case collars 20x34 + 2c, c = 0.1 / 0.2 / 0.3 (servo_pocket_clearance), 1/2/3 notches each.
"""
from __future__ import annotations

from build123d import Box, Cylinder, Pos

from params import P
from parts.common import Body, box

PR = P["print"]


def tolerance_piece() -> list[Body]:
    S = P.structure
    plate_t = 6.0
    rows = [
        (3.0, 3.2, 3.4),
        (3.4, 3.6, 3.8),
        (2.1, 2.3, 2.5),
        (1.5, 1.7, 1.9),
        (3.10, 3.25, 3.40),
    ]
    bores = [(S.ankle.bearing.D - 0.10, S.ankle.bearing.D - 0.05, S.ankle.bearing.D),
             (S.linkage.bearing.D - 0.10, S.linkage.bearing.D - 0.05, S.linkage.bearing.D),
             (S.wheel_set.clutch.D - 0.10, S.wheel_set.clutch.D - 0.05, S.wheel_set.clutch.D)]
    pitch = 10.0
    w, h = 4 * pitch + 10, (len(rows) + len(bores)) * pitch + 10
    plate = box(0, w, 0, h, 0, plate_t)
    r = 0
    for vals in rows + bores:
        r += 1
        y = h - r * pitch
        for i, d in enumerate(vals):
            plate = plate - Pos(10 + i * pitch * 1.3, y, plate_t / 2) * Cylinder(d / 2, plate_t + 2)
        for k in range(r):                                          # row number as edge notches
            plate = plate - box(-0.1, 1.2, y - 3 + k * 1.2, y - 2.4 + k * 1.2, -1, plate_t + 1)
    # horn recesses on a second plate
    horn = box(0, 3 * 22 + 4, 0, 24, 0, 4.0)
    for i, d in enumerate((16.0, 16.3, 16.6)):
        horn = horn - Pos(2 + 11 + i * 22, 12, 4.0 - 0.5) * Cylinder(d / 2, 1.0 + 0.01)
        horn = horn - Pos(2 + 11 + i * 22, 12, 2) * Cylinder(3.0, 5)
        for k in range(i + 1):
            horn = horn - box(2 + 3 + i * 22 + k * 1.5, 2 + 3.7 + i * 22 + k * 1.5, -0.1, 1.2, -1, 5)
    horn = Pos(w + 5, 0, 0) * horn
    # servo-case collars
    g = P.servo.geometry
    collars = None
    x = w + 5
    for i, c in enumerate((0.1, 0.2, 0.3)):
        iw, il = g.case[0] + 2 * c, g.case[1] + 2 * c
        col = Pos(x + iw / 2 + 2, 30 + il / 2 + 2, 2.5) * (Box(iw + 4, il + 4, 5) - Box(iw, il, 6))
        for k in range(i + 1):
            col = col - box(x + 3 + k * 1.5, x + 3.7 + k * 1.5, 30 - 0.1, 30 + 1.2, -1, 6)
        collars = col if collars is None else collars + col
        x += iw + 7
    return [Body("tolerance_holes", plate, "petg_structural", printed=True),
            Body("tolerance_horn", horn, "petg_structural", printed=True),
            Body("tolerance_servo_collars", collars, "petg_structural", printed=True)]
