#!/usr/bin/env python3
"""Write a gesture clip whose feet leave the floor, for a task to learn by imitation.

    python tools/synth_steps.py tasks/jumper/gesture_flash/media/flash.steps.json

`synth_gesture.py` interpolates joint keyframes and then solves the base from the
feet with `solve_resting_pose`, so every row is a pose the robot holds at rest. That
is right for a gesture done standing still, and it cannot write one that travels: a
tripod in the air mid-step is not resting on anything, and a body thrown sideways
over it is not at rest either. This tool works the other way round. The base's
trajectory and each foot's path over the floor are the choreography, and the joints
are solved from them row by row, with inverse kinematics on the robot's own model.

## The file

    {"move": "flash", "description": "one line, kept in the clip", "params": {}}

`move` names one of the choreographies in `MOVES`; `params` overrides any of its
numbers (the defaults are in `FLASH` and `SHAKE`, each one commented). The `.npz`
is written next to the file, named after the move, in the schema
`import_wbc_dances.py` writes: commanded and measured are the same track.

## How a move is built

A move is a list of segments. Each gives the base pose as a function of the
segment's phase, the arms' joints (the claws are the dancer's arms; a claw that is
standing on the floor is held there by IK like any other foot), and the steps taken
in it: which leg, between which phases, to which point on the floor. A swing is
interpolated across the floor in the base frame, from where the foot lifted to
where it lands -- interpolated in the world instead, a foot that lands where the
body is going cuts inward through the body's own sweep -- and raised over the floor
by the leg's `LIFT` on a half sine. Every other foot stays where it stands. A claw
that is `free` in a segment is in the air for all of it, and its joints come from
the arm function, except where they would put it through the floor.

## What is checked, and what is not

The worst leg-IK residual is printed, with the rows over 3 mm, and so are the
fastest joint, the base's travel, height, tilt and twist. The loader's checks run
when the task reads the clip: no joint faster than `CORNER_SPEED`, and the support
feet resting in one plane at the median row -- which is why a step here takes the
first half of its beat and not all of it. Whether physics can play the clip is not
this tool's question: a playback on the task's actuators and the training run are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

# `tools/` is not a package; run as a script, its own directory is on `sys.path`.
import import_wbc_dances as dances
import import_wbc_gestures as gestures
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

DT = 0.02
LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")
TRIPOD_A = ("LF", "RM", "LR")
TRIPOD_B = ("RF", "LM", "RR")
#: How high each foot swings, metres. The rear legs cannot lift much further while
#: the body is over them: at 15 mm their IK misses by several millimetres.
LIFT = {"LF": 0.02, "RF": 0.02, "LM": 0.015, "RM": 0.015, "LR": 0.012, "RR": 0.012}
#: A leg's joints, by the robot's own names. The claws have a fifth, the jaw, which
#: does not move the foot and is never solved for.
LEG_JOINTS = {
    **{leg: [f"{leg}_J{i}_joint" for i in range(4)] for leg in ("LF", "RF")},
    **{leg: [f"{leg}_J{i}_joint" for i in range(3)] for leg in ("LM", "RM", "LR", "RR")},
}
JAWS_SHUT = {"LF_J4_joint": 0.0, "RF_J4_joint": 0.0}


def smooth(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def lerp(a: dict, b: dict, u: float) -> dict:
    return {k: a[k] + (b[k] - a[k]) * u for k in a}


def tween(b0: np.ndarray, b1: np.ndarray) -> Callable[[float], np.ndarray]:
    """Base pose from b0 to b1 on a smoothstep."""
    b0, b1 = np.asarray(b0, float), np.asarray(b1, float)
    return lambda u: b0 + (b1 - b0) * smooth(u)


def frame_of(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rotation, origin) of base pose b = (x, y, z, roll, pitch, yaw), with
    R = Rz(yaw) Ry(pitch) Rx(roll) -- the schema's convention."""
    return Rotation.from_euler("xyz", b[3:]).as_matrix(), np.asarray(b[:3], float)


def support_margin(feet: np.ndarray, com: np.ndarray) -> float:
    """How far `com` is inside the convex hull of the `feet` (all [.., 2], on the
    floor), metres; negative outside it. Fewer than three feet hold nothing up."""
    if len(feet) < 3:
        return -np.inf
    hull = ConvexHull(feet)
    # Each facet is n.x + d <= 0 inside, with n a unit normal.
    return float(-(hull.equations[:, :2] @ com + hull.equations[:, 2]).max())


class Builder:
    """The clip, segment by segment: the base, the feet on the floor, the joints."""

    def __init__(self) -> None:
        from tasks.jumper.common.constants import HOME

        self.robot = gestures._Robot()
        self.names = self.robot.names
        self.lo, self.hi = self.robot.range[:, 0] + 1e-3, self.robot.range[:, 1] - 1e-3
        self.idx = {leg: [self.names.index(j) for j in js] for leg, js in LEG_JOINTS.items()}
        self.q = np.array([HOME[n] for n in self.names])
        feet = self._feet(self.q)
        # HOME, standing at the origin: the lowest foot on the floor.
        self.home = np.array([0.0, 0.0, self.robot.home_z - feet[:, 2].min(), 0.0, 0.0, 0.0])
        self.floor = self.robot.home_z
        self.local = {leg: feet[k] for k, leg in enumerate(LEGS)}
        self.feet = {leg: self.home[:3] + feet[k] for k, leg in enumerate(LEGS)}
        self.rows_q: list[np.ndarray] = []
        self.rows_b: list[np.ndarray] = []
        self.worst = 0.0
        self.missed: list[tuple[int, str, float, np.ndarray]] = []
        # The least the centre of mass is inside the support polygon, and its row.
        self.margin = (np.inf, -1)
        # The lowest a claw's joints alone would put it, against the floor: below
        # zero, `segment` lifted it back onto the floor.
        self.lowest = np.inf

    def _feet(self, q: np.ndarray) -> np.ndarray:
        return self.robot.feet_and_com(q[None])[0][0]

    def _ik(self, q: np.ndarray, leg: str, target: np.ndarray) -> tuple[np.ndarray, float]:
        """q with `leg`'s joints moved to put its foot at `target` (base frame)."""
        idx, k = self.idx[leg], LEGS.index(leg)

        def residual(x: np.ndarray) -> np.ndarray:
            qq = q.copy()
            qq[idx] = x
            # A whisper of regularisation keeps an unreachable target from
            # throwing the leg into a far branch.
            return np.concatenate([self._feet(qq)[k] - target, 1e-4 * (x - q[idx])])

        sol = least_squares(residual, q[idx], bounds=(self.lo[idx], self.hi[idx]))
        q = q.copy()
        q[idx] = sol.x
        return q, float(np.abs(residual(sol.x)[:3]).max())

    def landing(self, b: np.ndarray, leg: str, wider: float = 0.0) -> np.ndarray:
        """Where `leg` stands, on the floor, under base pose b: its HOME place, or
        `wider` metres further out to its own side."""
        rot, origin = frame_of(b)
        local = self.local[leg] + (0.0, np.sign(self.local[leg][1]) * wider, 0.0)
        w = rot @ local + origin
        w[2] = self.floor
        return w

    def claw_at(self, b: np.ndarray, leg: str, w: np.ndarray) -> dict:
        """The claw joints that put its foot at world point w under base pose b."""
        rot, origin = frame_of(b)
        q, _ = self._ik(self.q, leg, rot.T @ (w - origin))
        return {self.names[i]: float(q[i]) for i in self.idx[leg]}

    def claw_joints(self, leg: str) -> dict:
        """`leg`'s joints as they stand now, jaw shut."""
        return {**{self.names[i]: float(self.q[i]) for i in self.idx[leg]},
                f"{leg}_J4_joint": 0.0}

    def claw_reach(self, leg: str, tip: np.ndarray) -> dict:
        """The joints that put `leg`'s foot at `tip` in the base frame, from where the
        leg is now -- so a pose reached through a sequence stays on one branch."""
        q, miss = self._ik(self.q, leg, np.asarray(tip, float))
        if miss > 0.003:
            raise ValueError(f"{leg} cannot reach {np.round(tip, 3)}: misses by {miss * 1000:.1f} mm")
        return {self.names[i]: float(q[i]) for i in self.idx[leg]}

    def segment(
        self,
        seconds: float,
        base: Callable[[float], np.ndarray],
        arms: Callable[[float], dict],
        steps: list[tuple] = (),
        free: tuple[str, ...] = (),
    ) -> None:
        """`steps`: (leg, phase lifted, phase landed, world point landed on), and
        optionally how high it swings if not `LIFT`."""
        n = max(1, round(seconds / DT))
        plan = {}
        for leg, u0, u1, target, *lift in steps:
            r0, p0 = frame_of(base(u0))
            r1, p1 = frame_of(base(u1))
            plan[leg] = (u0, u1, r0.T @ (self.feet[leg] - p0), r1.T @ (target - p1), target,
                         lift[0] if lift else LIFT[leg])
        for k in range(1, n + 1):
            u = k / n
            b = base(u)
            rot, origin = frame_of(b)
            for leg, (u0, u1, lifted, landed, target, lift) in plan.items():
                if u0 < u < u1:
                    v = (u - u0) / (u1 - u0)
                    w = rot @ (lifted + (landed - lifted) * smooth(v)) + origin
                    # Over the floor, not over the body: a body that bobs down
                    # mid-step would carry a swing measured from it into the floor.
                    w[2] = self.floor + lift * np.sin(np.pi * v)
                    self.feet[leg] = w
                elif u >= u1:
                    self.feet[leg] = target.copy()
            q = self.q.copy()
            for name, value in arms(u).items():
                q[self.names.index(name)] = value
            for leg in LEGS:
                if leg in free:
                    continue
                target = rot.T @ (self.feet[leg] - origin)
                q, miss = self._ik(q, leg, target)
                self.worst = max(self.worst, miss)
                if miss > 0.003:
                    self.missed.append((len(self.rows_q), leg, miss, target))
            # A claw in the air goes where its joints put it, except through the
            # floor: there it is lifted onto it -- which happens as it sets down,
            # where the body's bob and shudder carry the last frames below it.
            for leg in free:
                w = rot @ self._feet(q)[LEGS.index(leg)] + origin
                self.lowest = min(self.lowest, w[2] - self.floor)
                if w[2] < self.floor - 5e-4:
                    w[2] = self.floor
                    q, miss = self._ik(q, leg, rot.T @ (w - origin))
                    self.worst = max(self.worst, miss)
            self.q = q
            # Whether the feet on the floor hold the body up: the centre of mass's
            # distance inside their polygon (negative: outside, toppling).
            on_floor = [leg for leg in LEGS if leg not in free
                        and not (leg in plan and plan[leg][0] < u < plan[leg][1])]
            feet, com = self.robot.feet_and_com(q[None])
            xy = (rot @ feet[0].T).T[[LEGS.index(leg) for leg in on_floor], :2] + origin[:2]
            margin = support_margin(xy, (rot @ com[0] + origin)[:2])
            if margin < self.margin[0]:
                self.margin = (margin, len(self.rows_q))
            self.rows_q.append(q.copy())
            self.rows_b.append(np.asarray(b, float).copy())


# ── The choreographies ───────────────────────────────────────────────────────

FLASH = {
    # Directions of the successive flashes, +1 to the robot's left (+y). The dancer
    # alternates, so the body drifts out and back.
    "order": [1, -1, 1, -1],
    # Seconds: standing still before each flash (the first one, and the rest), the
    # sway that winds it up, the flash out, the deepest point held, and the snap back.
    # The reference repeats every 1.6-1.7 s: 0.1 + 0.18 + 0.12 + 0.18 out and back,
    # two follow steps of 0.15 (`follow_s`), and 0.75 still.
    "first_still_s": 0.6,
    "still_s": 0.75,
    "sway_s": 0.10,
    "out_s": 0.18,
    "peak_s": 0.12,
    "back_s": 0.18,
    # The sway: this far the other way, and this much lower.
    "counter": 0.006,
    "dip": 0.004,
    # At the deepest point: the base this low (HOME stands at 107 mm; the dancer's
    # hips drop 12-15%), shifted this far toward the flash, leaned away from it
    # (roll, over the planted side), pitched forward, and twisted against the claw
    # that is flung (yaw), radians. The dancer leans 20-25 degrees; with the pitch
    # this roll tilts the body 20.6. The legs planted on the low side reach at any
    # lean up to 0.38; what limits it is the middle leg shooting out on the high
    # side (`wide`). At 0.17 the trained policy leaned 10.4 degrees, half the
    # dancer's.
    "peak_z": 0.090,
    "peak_shift": 0.025,
    "roll": 0.36,
    "pitch": 0.03,
    "yaw": 0.20,
    # The middle leg on the flash side shoots out sideways and slightly back from
    # its place under the deepest pose, metres. The lean already carries that place
    # 22 mm further out, so on the floor the foot moves about 58 mm. The higher
    # the body rolls that side the less floor the leg reaches: at 9.7 degrees 55 mm
    # beyond its place, at 16.6 35, at 20.6 25 -- beyond that it misses its IK,
    # flips branch and whips its knee past the servo's corner speed. 10 mm short of
    # the limit its knee peaks at 17.8 rad/s; 2.5 mm short of it, at 27.
    "wide": 0.015,
    "back": 0.02,
    # How high it lifts on the way out and back, metres. At 8 mm the trained
    # policy's foot rose at most 16 mm: a slide no one could see.
    "skim": 0.03,
    # Where the body stands after each flash: this far toward it (the dancer moves
    # 0.3-0.4 shoulder widths). The snap back carries it `back_share` of the way;
    # the four legs that did not step then follow in two quick diagonal pairs of
    # `follow_s` each, carrying it the rest. Without them the body can go no
    # further than about 40 mm: the legs left behind run out of reach.
    "drift": 0.08,
    "back_share": 0.5,
    "follow_s": 0.15,
    # How high those follow steps lift, metres. At `LIFT`'s 12-20 mm the trained
    # policy's feet rose at most 16 mm and the scuttle could barely be seen; at 30
    # the rear legs miss their IK by 3-4 mm, at 25 by under 2.
    "follow_lift": 0.025,
    # The claw on the flash side, flung out low and wide: its tip in the base frame
    # (left claw; the right one mirrored), metres.
    "claw_tip": [0.09, 0.27, -0.025],
    # Its jaw, opened while it is out (the left claw's sign).
    "jaw": -0.8,
}


def flash(p: dict) -> Builder:
    """The flash step, as the reference does it: standing still, a sway, then in a
    fifth of a second the middle leg on one side shoots out sideways, the body drops
    and leans away over the planted legs, twisting, while that side's claw is flung
    out low and wide; a beat held, then everything snaps back to standing and the
    legs left behind scuttle after it, the body 8 cm over. Still again, then the
    other way."""
    m = Builder()
    shut = lambda u: JAWS_SHUT
    b = m.home.copy()
    for k, s in enumerate(p["order"]):
        m.segment(p["first_still_s"] if k == 0 else p["still_s"], lambda u, b=b: b, shut)
        leg, claw = ("LM", "LF") if s > 0 else ("RM", "RF")
        sway = b.copy()
        sway[1] -= s * p["counter"]
        sway[2] -= p["dip"]
        peak = b.copy()
        peak[1] += s * p["peak_shift"]
        peak[2] = p["peak_z"]
        peak[3] = s * p["roll"]
        peak[4] = p["pitch"]
        peak[5] = -s * p["yaw"]
        rest = m.home.copy()
        rest[:2] = b[:2]
        rest[1] += s * p["drift"]
        # Where the snap back leaves the body, and where the first follow step does:
        # the legs left behind cannot reach `drift` from where they stand.
        back = rest.copy()
        back[1] -= s * p["drift"] * (1.0 - p["back_share"])
        half = rest.copy()
        half[1] -= s * p["drift"] * (1.0 - p["back_share"]) / 2.0
        out = m.landing(peak, leg)
        out[0] -= p["back"]
        out[1] += s * p["wide"]

        down = m.claw_joints(claw)
        tip = np.array(p["claw_tip"]) * (1, s, 1)
        # The jaws' ranges are mirror images: the left opens negative, the right positive.
        flung = {**m.claw_reach(claw, tip), f"{claw}_J4_joint": s * p["jaw"]}
        landed = m.landing(rest, claw)
        home = {**m.claw_at(rest, claw, landed), f"{claw}_J4_joint": 0.0}

        def arm(a: dict, z: dict, u0: float, u1: float) -> Callable[[float], dict]:
            return lambda u: {**JAWS_SHUT, **lerp(a, z, u0 + (u1 - u0) * smooth(u))}

        m.segment(p["sway_s"], tween(b, sway), arm(down, flung, 0.0, 0.2), free=(claw,))
        m.segment(p["out_s"], tween(sway, peak), arm(down, flung, 0.2, 1.0),
                  [(leg, 0.0, 1.0, out, p["skim"])], free=(claw,))
        m.segment(p["peak_s"], lambda u, peak=peak: peak, arm(flung, flung, 0.0, 0.0),
                  free=(claw,))
        m.segment(p["back_s"], tween(peak, back), arm(flung, home, 0.0, 1.0),
                  [(leg, 0.0, 1.0, m.landing(rest, leg), p["skim"])], free=(claw,))
        m.feet[claw] = landed
        # The four legs that stayed behind follow in two diagonal pairs, so that
        # four feet stand under the body through each: the scuttle that carries
        # the crab over.
        near, far = ("LR", "RM") if s > 0 else ("RR", "LM")
        first = (near, far)
        second = ("RF", "RR") if s > 0 else ("LF", "LR")
        m.segment(p["follow_s"], tween(back, half), shut,
                  [(x, 0.0, 1.0, m.landing(rest, x), p["follow_lift"]) for x in first])
        m.segment(p["follow_s"], tween(half, rest), shut,
                  [(x, 0.0, 1.0, m.landing(rest, x), p["follow_lift"]) for x in second])
        b = rest
    m.segment(p["still_s"], lambda u: b, shut)
    return m


SHAKE = {
    # Base height while dancing: a deep horse stance (HOME stands at 107 mm; the
    # dancer's hips are 20-25% down; 12% is as deep as the legs stay inside their
    # reach while the body bobs, leans and travels over them).
    "crouch_z": 0.094,
    # Seconds to get down into it, both tripods stepping out on the way.
    "stance_s": 0.6,
    # Phrases: (seconds a beat, beats, +1 travelling to the robot's left / -1 back).
    # The dancer takes a new stance every 0.25-0.5 s, about two steps a second, and
    # crosses the stage and back; the second phrase is quicker, as theirs builds.
    # The claws' swings set the pace: at 0.40 / 0.36 s the feet must hold a yaw
    # torque of 808 mN*m (95th percentile, kinematic), more than the 688 that left
    # an earlier shake untrainable; at 0.45 / 0.42 it is 601.
    "phrases": [[0.45, 7, 1], [0.40, 7, -1]],
    # Metres the body travels sideways each beat, and forward on the left claw's
    # beats and back on the right's: the cross step's zigzag. Seven beats of 30 mm
    # carry it 210 mm, two thirds of its width; the dancer crosses 1-1.5 of theirs.
    "step": 0.03,
    "zigzag": 0.012,
    # Feet land this many beats' travel ahead of where the body ends the beat:
    # a foot stays planted from the middle of one of its tripod's beats to the
    # start of the next, and lands near the middle of that.
    "lead": 0.3,
    # How high the stepping legs lift, metres: high enough to see from across the
    # room -- the dancer stamps into every new stance. At 8 mm the trained policy's
    # feet rose 10-17 mm and the steps read as a shuffle. Above 30 mm the rear legs,
    # twisted under the body mid-beat, miss by more than 5 mm at the top of the swing.
    "lift": 0.030,
    # The stance is a wide one, as the dancer's: every foot lands this much further
    # out to its side than at HOME, which also leaves the folded legs room to lift.
    "wider": 0.020,
    # Knees down this far on every beat, on a squared sine.
    "bob": 0.008,
    # Twisted (yaw) and leaned (roll) towards the side the claw is flung to, radians.
    "yaw": 0.12,
    "roll": 0.06,
    # The shudder over everything: the body's height and roll, and the flung
    # claw's shoulder, at this rate (the dancer's is 4-6 Hz).
    "tremble_hz": 5.0,
    # At 3 mm the trained policy shuddered 3.4 mm (95th percentile): too small to see.
    "tremble_z": 0.007,
    "tremble_roll": 0.03,
    "claw_tremble": 0.08,
    # The bob and the shudder grow by this factor from the first beat to the last.
    "grow": 1.3,
    # Where a flung claw's tip goes, in the base frame (left claw; the right one
    # mirrored), metres, in the dancer's order: up overhead in a V, crossed in front,
    # flung out flat. The claws take turns, one shape a pair of beats.
    "shapes": [[0.07, 0.15, 0.12], [0.17, -0.02, 0.04], [0.10, 0.26, 0.0]],
    # Which way the body twists for each shape: towards the claw (+1), or with it
    # as it crosses (-1). The dancer's arms lead the twist.
    "shape_twist": [1, -1, 1],
    # The final pose, both claws raised (the dancer's W): the tips, the base shifted
    # back over the four legs first, its height, and seconds to get there, to hold
    # it, and to come down. The claws are lightest on the floor at the end of
    # coming down; there, shifted back 25 mm the centre of mass is 0.3 mm inside
    # the four feet and a playback on the actuators tips forward, at 50 mm 18 mm.
    "w_tip": [0.06, 0.14, 0.13],
    "w_back": 0.05,
    "w_z": 0.104,
    "w_s": 0.35,
    "w_hold_s": 0.7,
}

#: The two legs that step with each claw: its tripod.
TRIPOD_OF = {"LF": ("RM", "LR"), "RF": ("LM", "RR")}


def shake(p: dict) -> Builder:
    """The bear shake, as the reference does it: a deep stance, a new step every
    beat, travelling sideways and back, with the arms swinging big -- up overhead,
    crossed in front, flung out flat -- and the whole body shuddering; it ends with
    both arms raised. The crab steps in tripods so that one is always standing, and
    its claws take turns as the arms: every beat one tripod swings, its claw flung
    into the next shape while its two legs step, the body bobbing, twisting with
    the claw and leaning into it."""
    m = Builder()
    shut = lambda u: JAWS_SHUT
    b = m.home.copy()
    m.segment(0.3, lambda u: b, shut)
    ready = b.copy()
    ready[2] = p["crouch_z"]
    # Down into the stance while both tripods step out wide, as the dancer drops into
    # the horse stance before the first beat. Started from HOME's narrower stance,
    # the left rear foot ends the first beat 45 mm under the body, which has twisted
    # and travelled over it, and cannot lift from there.
    m.segment(p["stance_s"], tween(b, ready), shut,
              [(leg, 0.0, 0.5, m.landing(ready, leg, p["wider"])) for leg in TRIPOD_A]
              + [(leg, 0.5, 1.0, m.landing(ready, leg, p["wider"])) for leg in TRIPOD_B])
    b = ready.copy()
    total = sum(n for _, n, _ in p["phrases"])
    k, t = 0, 0.0
    for seconds, n, direction in p["phrases"]:
        for _ in range(n):
            # The right tripod first: travelling left, it carries the left middle leg,
            # which the body would otherwise come down on top of.
            claw = "RF" if k % 2 == 0 else "LF"
            side = 1 if claw == "LF" else -1
            shape = (k // 2) % len(p["shapes"])
            g = 1.0 + (p["grow"] - 1.0) * k / (total - 1)
            nb = b.copy()
            nb[0] = ready[0] + side * p["zigzag"]
            nb[1] += direction * p["step"]
            nb[3] = -side * p["roll"]
            nb[5] = p["shape_twist"][shape] * side * p["yaw"]
            line = tween(b, nb)

            def base(u: float, line=line, g=g, t0=t, seconds=seconds) -> np.ndarray:
                x = line(u).copy()
                phase = 2 * np.pi * p["tremble_hz"] * (t0 + u * seconds)
                x[2] = (p["crouch_z"] - p["bob"] * g * np.sin(np.pi * u) ** 2
                        + p["tremble_z"] * g * np.sin(phase))
                x[3] += p["tremble_roll"] * g * np.sin(phase + 1.0)
                return x

            # Feet land under a square body a little further along the phrase, not
            # under the twisted, leaning one: planted there, a foot stays inside
            # its reach while the body twists the other way and travels on over it.
            square = ready.copy()
            square[1] = nb[1] + direction * p["lead"] * p["step"]
            start = m.claw_joints(claw)
            flung = m.claw_reach(claw, np.array(p["shapes"][shape]) * (1, side, 1))
            landed = m.landing(square, claw, p["wider"])
            end = {**m.claw_at(nb, claw, landed), f"{claw}_J4_joint": 0.0}
            flung[f"{claw}_J4_joint"] = side * -0.8  # the jaw opens as it flies

            def arms(u: float, start=start, end=end, flung=flung, claw=claw, g=g, t0=t,
                     seconds=seconds) -> dict:
                w = np.sin(np.pi * u / 0.9) ** 2 if u < 0.9 else 0.0
                j = lerp(lerp(start, end, smooth(u / 0.95)), flung, w)
                phase = 2 * np.pi * p["tremble_hz"] * (t0 + u * seconds)
                j[f"{claw}_J1_joint"] += p["claw_tremble"] * g * w * np.sin(phase)
                return {**JAWS_SHUT, **j}

            m.segment(seconds, base, arms,
                      [(leg, 0.05, 0.5, m.landing(square, leg, p["wider"]), p["lift"])
                       for leg in TRIPOD_OF[claw]],
                      free=(claw,))
            m.feet[claw] = landed
            b = nb
            k += 1
            t += seconds
    # Square up and re-plant both tripods where the travel ended.
    settle = ready.copy()
    settle[:2] = (ready[0], b[1])
    m.segment(0.4, tween(b, settle), shut)
    m.segment(0.6, lambda u: settle, shut,
              [(leg, 0.0, 0.5, m.landing(settle, leg)) for leg in TRIPOD_A]
              + [(leg, 0.5, 1.0, m.landing(settle, leg)) for leg in TRIPOD_B])
    # The W: the body back over the four legs while the claws still stand, then
    # both claws up, held, and down again.
    lean = settle.copy()
    lean[0] -= p["w_back"]
    lean[2] = p["w_z"]
    m.segment(0.3, tween(settle, lean), shut)
    down = {**m.claw_joints("LF"), **m.claw_joints("RF")}
    up = {**m.claw_reach("LF", np.array(p["w_tip"])),
          **m.claw_reach("RF", np.array(p["w_tip"]) * (1, -1, 1)),
          "LF_J4_joint": -0.8, "RF_J4_joint": 0.8}
    m.segment(p["w_s"], lambda u: lean, lambda u: lerp(down, up, smooth(u)), free=("LF", "RF"))
    m.segment(p["w_hold_s"], lambda u: lean, lambda u: up, free=("LF", "RF"))
    m.segment(0.4, lambda u: lean, lambda u: lerp(up, down, smooth(u)), free=("LF", "RF"))
    stand = m.home.copy()
    stand[:2] = settle[:2]
    m.segment(0.4, tween(lean, stand), shut)
    m.segment(0.3, lambda u: stand, shut)
    return m


SHAKE_TROT = {
    # The stance: crouched this low, every foot this much wider than HOME, stepped
    # into over this many seconds.
    "crouch_z": 0.097,
    "wider": 0.025,
    "rear_wider": 0.025,
    "stance_s": 0.6,
    # The four legs trot: a diagonal pair in the air at a time, the other pair's
    # line under the centre of mass. The pairs' lines cross midway between the
    # middle and rear feet; the rear feet stand this much further forward than
    # their places, so that the body need not go as far back to put the crossing
    # under the centre of mass.
    "rear_in": 0.04,
    # The claws stand this much closer in, so that they still reach the floor with
    # the body back over the four legs.
    "claw_in": 0.04,
    "lean_s": 0.5,
    # Seconds raising both claws, slowly: the claws are 27% of the robot, and swung
    # up in 0.35 s their reaction pitches the body onto its face.
    "raise_s": 1.0,
    # Seconds moving the body over the first diagonal pair's line. Every row after
    # that puts the centre of mass over the line of the pair on the floor, the
    # claws' share of it included, and moves it to the next pair's line while all
    # four feet are down.
    "shift_s": 0.4,
    # A pair's beat: in the air, then all four down.
    "swing_s": 0.22,
    "double_s": 0.06,
    # Cycles (+1 travelling to the robot's left, -1 back), each both pairs, and the
    # metres each one carries the body; feet land this share of a stride ahead.
    "cycles": [1, 1, 1, 1, -1, -1, -1, -1],
    "stride": 0.045,
    "lead": 0.3,
    "lift": 0.030,
    "rear_lift": 0.027,
    # Knees down on every beat; the shudder over everything. No roll in it: the
    # body is on a line for most of the time, and a roll would tip it over the line.
    "bob": 0.006,
    "tremble_hz": 5.0,
    "tremble_z": 0.006,
    "tremble_roll": 0.0,
    # The claws shudder in opposite directions, so their reactions cancel in pitch.
    "claw_tremble": 0.06,
    # The claws keep their own time: a new shape every this many seconds, reached
    # over this share of it and held for the rest. The dancer changes every
    # 0.25-0.5 s; the claws are 27% of the robot, so they take the slow end.
    "shape_s": 0.48,
    "shape_share": 0.7,
    # The claws' shapes, as tips in the base frame, metres: (left, right), or the
    # left alone and mirrored. The dancer's arms hold one shape for a quarter of a
    # second: forward-down crossing the hips, out low, up in a V, crossed in front
    # of the chest, flung out flat, ...; and every second or so both swing to the
    # same side, one across, one flung out, with the torso leaning into it.
    "shapes": {
        "out_low": [0.12, 0.24, -0.02],
        "V": [0.07, 0.15, 0.12],
        "chest": [0.19, 0.06, 0.05],
        "T": [0.10, 0.26, 0.0],
        "up_left": [[0.07, 0.15, 0.13], [0.09, -0.20, 0.07]],
        "swing_left": [[0.08, 0.27, 0.02], [0.19, 0.03, 0.05]],
        "wide_left": [[0.10, 0.24, 0.07], [0.11, -0.24, 0.0]],
        "up_right": [[0.09, 0.20, 0.07], [0.07, -0.15, 0.13]],
        "swing_right": [[0.19, -0.03, 0.05], [0.08, -0.27, 0.02]],
        "wide_right": [[0.11, 0.24, 0.0], [0.10, -0.24, 0.07]],
        "W": [0.06, 0.14, 0.13],
    },
    # The shapes in order, repeating; the swings turn the body their way.
    "order": ["V", "chest", "T", "up_left", "swing_left", "wide_left", "V", "out_low",
              "V", "chest", "T", "up_right", "swing_right", "wide_right", "V", "out_low"],
    # The body turns this far towards a swing, radians. The dancer leans into it
    # too, 30 degrees; the crab cannot: standing on a line between two feet, a lean
    # tips it over the line.
    "lunge_yaw": 0.03,
    # Seconds on all four feet where the travel turns back.
    "turn_s": 0.6,
    "w_hold_s": 0.7,
}


class _ClawsUp:
    """What the bear shakes with both claws up share: the stance, the claws' dance, the
    centre of mass held over the four legs, and the way in and out of it all."""

    def __init__(self, p: dict) -> None:
        self.p = p
        self.m = Builder()
        self.claws = ("LF", "RF")
        self.rear = ("LR", "RR")
        self.poses: dict[str, dict] = {}

    def place(self, b: np.ndarray, leg: str) -> np.ndarray:
        """Where `leg` stands in the stance under base pose b: out to its side, the rear
        feet further forward and the claws further in."""
        p, m = self.p, self.m
        w = m.landing(b, leg, p["rear_wider"] if leg in self.rear else p["wider"])
        shift = p["rear_in"] if leg in self.rear else -p["claw_in"] if leg in self.claws else 0.0
        rot, _ = frame_of(b)
        w = w + rot @ np.array([shift, 0.0, 0.0])
        w[2] = m.floor
        return w

    def timeline(self, t: float) -> dict:
        p, order = self.p, self.p["order"]
        k = int(t // p["shape_s"])
        u = t / p["shape_s"] - k
        return lerp(self.poses[order[k % len(order)]], self.poses[order[(k + 1) % len(order)]],
                    smooth(u / p["shape_share"]))

    def side_at(self, t: float) -> int:
        order = self.p["order"]
        name = order[(int(t // self.p["shape_s"]) + 1) % len(order)]
        return 1 if "left" in name else -1 if "right" in name else 0

    def dance(self, t: float) -> dict:
        p, m = self.p, self.m
        j = self.timeline(t)
        wobble = p["claw_tremble"] * np.sin(2 * np.pi * p["tremble_hz"] * t)
        for c, sign in (("LF", 1), ("RF", -1)):
            k = m.names.index(f"{c}_J1_joint")
            j[f"{c}_J1_joint"] = float(np.clip(j[f"{c}_J1_joint"] + sign * wobble, m.lo[k], m.hi[k]))
        return {**JAWS_SHUT, **j}

    def com_offset(self, b: np.ndarray, t: float, arms: dict | None = None,
                   legs: np.ndarray | None = None) -> np.ndarray:
        """The centre of mass's horizontal offset from the base, the claws as the
        dance has them at t (or as `arms` has them) and the legs as they last stood
        (or as `legs` has them)."""
        m = self.m
        q = (m.q if legs is None else legs).copy()
        for name, value in (self.dance(t) if arms is None else arms).items():
            q[m.names.index(name)] = value
        rot, _ = frame_of(b)
        return (rot @ m.robot.feet_and_com(q[None])[1][0])[:2]

    def standing(self, b: np.ndarray) -> np.ndarray:
        """The joints with every foot where it stands, under base pose b."""
        m = self.m
        rot, origin = frame_of(b)
        q = m.q.copy()
        for leg in LEGS:
            q, _ = m._ik(q, leg, rot.T @ (m.feet[leg] - origin))
        return q

    def centred(self, b: np.ndarray, arms: dict) -> float:
        """The base x that puts the centre of mass midway between the middle and
        the rear feet, where the two pairs' lines cross: on four soft legs (kp 10)
        a body loaded onto one pair sags on that side and pitches over it."""
        m = self.m
        mid = np.mean([m.feet[leg][0] for leg in ("LM", "RM", "LR", "RR")])
        x = b.copy()
        for _ in range(3):
            x[0] = mid - self.com_offset(x, 0.0, arms, self.standing(x))[0]
        return float(x[0])

    def intro(self) -> tuple[np.ndarray, np.ndarray]:
        """Down into the stance, back over the four legs with the claws still on the
        floor, then both claws up, slowly, the body following the centre of mass as
        the claws carry theirs back. Returns (the crouched stance's base pose, the
        base pose with the claws up)."""
        p, m, claws = self.p, self.m, self.claws
        shut = lambda u: JAWS_SHUT
        b = m.home.copy()
        m.segment(0.3, lambda u: b, shut)
        ready = b.copy()
        ready[2] = p["crouch_z"]
        m.segment(p["stance_s"], tween(b, ready), shut,
                  [(leg, 0.0, 0.5, self.place(ready, leg)) for leg in TRIPOD_A]
                  + [(leg, 0.5, 1.0, self.place(ready, leg)) for leg in TRIPOD_B])
        q = m.q.copy()
        for name in [*p["order"], "W"]:
            if name in self.poses:
                continue
            tips = p["shapes"][name]
            left, right = (tips if isinstance(tips[0], list)
                           else (tips, [tips[0], -tips[1], tips[2]]))
            for leg, tip in (("LF", left), ("RF", right)):
                q, miss = m._ik(q, leg, np.array(tip, float))
                if miss > 0.003:
                    raise ValueError(f"{name}: {leg} cannot reach {tip}: misses by {miss * 1000:.1f} mm")
            self.poses[name] = {**{m.names[i]: float(q[i]) for c in claws for i in m.idx[c]},
                                **JAWS_SHUT}
        first = self.dance(0.0)
        lean = ready.copy()
        lean[0] = self.centred(ready, {})
        m.segment(p["lean_s"], tween(ready, lean), shut)
        down = {**m.claw_joints("LF"), **m.claw_joints("RF")}

        def rising(u: float) -> dict:
            return {**JAWS_SHUT, **lerp(down, first, smooth(u))}

        def raised(u: float) -> np.ndarray:
            x = lean.copy()
            x[0] = self.centred(x, rising(u))
            return x

        m.segment(p["raise_s"], raised, rising, free=claws)
        return ready, raised(1.0)

    def outro(self, b: np.ndarray, t: float, lean: np.ndarray) -> None:
        """Square up over the four legs, the claws into the W, held, and down -- as
        slowly as they went up, the body following the centre of mass forward; then
        back onto all six feet and up to HOME."""
        p, m, claws = self.p, self.m, self.claws
        w = {**self.poses["W"], "LF_J4_joint": -0.8, "RF_J4_joint": 0.8}
        settle = lean.copy()
        settle[1] = b[1]
        settle[0] = self.centred(settle, w)
        start = self.dance(t)
        m.segment(0.5, tween(b, settle), lambda u: lerp(start, w, smooth(u)), free=claws)
        m.segment(p["w_hold_s"], lambda u: settle, lambda u: w, free=claws)
        final = settle.copy()
        for _ in range(2):
            spots = {c: self.place(final, c) for c in claws}
            floor = {}
            for c in claws:
                floor.update({**m.claw_at(final, c, spots[c]), f"{c}_J4_joint": 0.0})
            final[0] = self.centred(final, floor)

        def lowering(u: float) -> dict:
            return lerp(w, floor, smooth(u))

        def lowered(u: float) -> np.ndarray:
            x = settle.copy()
            x[0] = self.centred(x, lowering(u))
            return x

        m.segment(p["raise_s"], lowered, lowering, free=claws)
        for c in claws:
            m.feet[c] = spots[c]
        settle = lowered(1.0)
        stand = m.home.copy()
        stand[1] = settle[1]
        shut = lambda u: JAWS_SHUT
        m.segment(0.5, tween(settle, stand), shut,
                  [(leg, 0.0, 0.5, m.landing(stand, leg)) for leg in TRIPOD_A]
                  + [(leg, 0.5, 1.0, m.landing(stand, leg)) for leg in TRIPOD_B])
        m.segment(0.3, lambda u: stand, shut)


def shake_trot(p: dict) -> Builder:
    """The bear shake with both claws up from the first beat to the last, the four
    back legs trotting sideways in diagonal pairs. (gesture_shake v8: it trained, and
    the policy shuffled its middle feet rather than stand on a diagonal pair.)"""
    rig = _ClawsUp(p)
    m, claws, rear = rig.m, rig.claws, rig.rear
    dance, side_at, com_offset = rig.dance, rig.side_at, rig.com_offset

    def shudder(x: np.ndarray, t: float, dip: float) -> np.ndarray:
        ph = 2 * np.pi * p["tremble_hz"] * t
        x[2] = p["crouch_z"] - dip + p["tremble_z"] * np.sin(ph)
        x[3] += p["tremble_roll"] * np.sin(ph + 1.0)
        return x

    def on_line(b: np.ndarray, t: float, pair: tuple[np.ndarray, np.ndarray]) -> float:
        """The base x that puts the centre of mass over the line between two feet,
        the base's y as it is."""
        a, e = pair
        c = com_offset(b, t)
        s = (b[1] + c[1] - a[1]) / (e[1] - a[1])
        return float(a[0] + s * (e[0] - a[0]) - c[0])

    ready, lean = rig.intro()
    seconds = p["swing_s"] + p["double_s"]
    land = p["swing_s"] / seconds
    plan = []
    for d in p["cycles"]:
        plan += [(("LM", "RR"), d), (("RM", "LR"), d)]
    pairs = {("LM", "RR"): ("RM", "LR"), ("RM", "LR"): ("LM", "RR")}
    # Onto the first pair's line, the claws dancing from here on.
    first_line = tuple(m.feet[leg].copy() for leg in pairs[plan[0][0]])
    start = lean.copy()
    goal = lean.copy()
    goal[0] = on_line(goal, p["shift_s"], first_line)
    t = 0.0
    m.segment(p["shift_s"], lambda u: shudder(tween(start, goal)(u).copy(), u * p["shift_s"], 0.0),
              lambda u: dance(u * p["shift_s"]), free=claws)
    t += p["shift_s"]
    b = goal.copy()
    for i, (pair, d) in enumerate(plan):
        nb = b.copy()
        nb[1] += d * p["stride"] / 2
        nb[5] = side_at(t + seconds) * p["lunge_yaw"]
        line = tween(b, nb)
        square = ready.copy()
        square[1] = nb[1] + d * p["lead"] * p["stride"]
        targets = {leg: rig.place(square, leg) for leg in pair}
        now = tuple(m.feet[leg].copy() for leg in pairs[pair])
        then = tuple(targets[leg] for leg in pair)

        def base(u: float, line=line, t0=t, now=now, then=then) -> np.ndarray:
            x = line(u).copy()
            tt = t0 + u * seconds
            x[0] = on_line(x, tt, now)
            if u > land:
                x[0] += (on_line(x, tt, then) - x[0]) * smooth((u - land) / (1 - land))
            return shudder(x, tt, p["bob"] * np.sin(np.pi * u) ** 2)

        steps = [(leg, 0.0, land, targets[leg], p["rear_lift"] if leg in rear else p["lift"])
                 for leg in pair]
        m.segment(seconds, base, lambda u, t0=t: dance(t0 + u * seconds), steps, free=claws)
        b = base(1.0)
        b[2], b[3] = p["crouch_z"], 0.0
        t += seconds
        if i + 1 < len(plan) and plan[i + 1][1] != d:
            # Turning back: all four feet down while the knees bob twice and the
            # claws dance on, the body kept over the next pair's line.
            def base(u: float, b=b, t0=t, then=then) -> np.ndarray:
                x = b.copy()
                tt = t0 + u * p["turn_s"]
                x[0] = on_line(x, tt, then)
                return shudder(x, tt, p["bob"] * np.sin(2 * np.pi * u) ** 2)

            m.segment(p["turn_s"], base, lambda u, t0=t: dance(t0 + u * p["turn_s"]), free=claws)
            b = base(1.0)
            b[2], b[3] = p["crouch_z"], 0.0
            t += p["turn_s"]
    rig.outro(b, t, lean)
    return m


SHAKE_CRAWL = {
    # The stance, crouched this low and stepped into over this many seconds. The
    # middle feet stand this much further out than HOME, the rear feet this much: with
    # both claws up the body leans back ~90 mm over the four legs (the claws are 28%
    # of the robot, held in front), which sweeps every foot forward in the body's
    # frame, and swept that far forward a rear leg cannot lift 30 mm unless its foot
    # is at least 20 mm further out than HOME -- nor a middle leg reach a stride out
    # past 45 mm. Wider still, the rear legs only sag further on the servos (kp 10),
    # and the middle legs' triangles leave the centre of mass no room 18 mm in. At 97 mm
    # (v9) the trailing middle foot, 25 mm in under the body while the rear pair steps,
    # could not fold that far back once the centre of mass went 18 mm in: lower, it can.
    # At 87 mm the body averages 20% below HOME through the dance and dips to 30% (the
    # dancer's horse stance drops 20-25%); at 85 the rear legs miss their swing by 5 mm.
    "crouch_z": 0.087,
    "wider": 0.01,
    "rear_wider": 0.04,
    "stance_s": 0.6,
    # The rear feet stand this much further forward than their places, the claws this
    # much closer in; as SHAKE_TROT's, and why.
    "rear_in": 0.04,
    "claw_in": 0.04,
    "lean_s": 0.5,
    "raise_s": 1.0,
    # The crawl: one leg in the air at a time, the other three holding the centre of
    # mass inside their triangle -- at least this far inside it, metres, for the whole
    # swing. The two middle legs lift with the centre of mass behind the diagonal
    # between the other middle foot and the opposite rear foot, the two rear legs with
    # it in front of the other diagonal. 8 mm (v9) was less than the training's own
    # randomised offset of the centre of mass (+/-8.7 mm fore and aft): its policy kept
    # the centre of mass forward and the middle feet down. 18 mm is as far in as the
    # middle legs reach: at 20 the triangle has no room left where the claws hold it.
    "margin": 0.018,
    # How hard the centre of mass is drawn back to the diagonals' crossing between
    # swings, against the acceleration it costs, 1/s^2: enough that it does not wander
    # to a corner of the four feet, little enough that the swings set the path.
    "centre_pull": 2.0,
    # Seconds on four feet before the middle pair steps (the body carried sideways a
    # stride, to midway between where the feet stood and where they will land, so
    # that every swing is centred under it) and before the rear pair.
    "shift_s": 0.28,
    "shift2_s": 0.28,
    # A foot's swing, and the four feet down between the two of a pair.
    "swing_s": 0.22,
    "double_s": 0.06,
    # Cycles (+1 to the robot's left, -1 back), each foot stepping once in each, the
    # leading foot of a pair first; the metres each carries the body; how high the
    # feet lift.
    "cycles": [1, 1, 1, 1, -1, -1, -1, -1],
    "stride": 0.05,
    "lift": 0.030,
    "rear_lift": 0.030,
    # Knees down on every swing; the shudder over everything, faded in and out over
    # half a second at the ends.
    "bob": 0.006,
    "tremble_hz": 5.0,
    "tremble_z": 0.006,
    # The claws as SHAKE_TROT's: the same shapes, order and timing.
    "claw_tremble": 0.06,
    "shape_s": 0.48,
    "shape_share": 0.7,
    "shapes": SHAKE_TROT["shapes"],
    "order": SHAKE_TROT["order"],
    # Seconds on four feet where the travel turns back, squaring up at the end, and
    # holding the W.
    "turn_s": 0.6,
    "square_s": 0.4,
    "w_hold_s": 0.7,
}


def shake_crawl(p: dict) -> Builder:
    """The bear shake with both claws up from the first beat to the last, the four
    back legs crawling sideways one at a time with the centre of mass inside the
    three feet on the floor.

    The trot before it (`shake_trot`) stood on two feet for 0.22 s of every 0.28: on
    the clip's own inverse dynamics the feet could not supply the moment it asked for
    in 14% of its rows, 0.095 N*m at the 95th percentile, and the policy trained on it
    kept its middle feet on the floor (17 mm lifts, 9 cm of travel). This one needs
    nothing the feet cannot give outside a touchdown's row or two -- the tripod
    clip that trained (`shake_tripod`) is the same."""
    rig = _ClawsUp(p)
    m, claws, rear = rig.m, rig.claws, rig.rear
    four = ("LM", "RM", "LR", "RR")
    ready, lean = rig.intro()

    def xy(legs: tuple[str, ...], feet: dict | None = None) -> np.ndarray:
        return np.array([(m.feet if feet is None else feet)[leg][:2] for leg in legs])

    def interval(support: tuple[str, ...], y: float,
                 feet: dict | None = None) -> tuple[float, float] | None:
        """The x range, at y, of the triangle of `support` feet (where they stand now,
        or in `feet`) inset by the margin."""
        lo, hi = -np.inf, np.inf
        for a, e, c in ConvexHull(xy(support, feet)).equations:  # a x + e y + c <= -margin
            r = -p["margin"] - c - e * y
            if abs(a) < 1e-9:
                if r < 0:
                    return None
            elif a > 0:
                hi = min(hi, r / a)
            else:
                lo = max(lo, r / a)
        return (lo, hi) if lo <= hi else None

    # The plan first -- every phase's length, the base's y, the step in it and the
    # three feet holding the body up -- so that the centre of mass's fore-and-aft swing
    # can be laid over all of it. Moved only in the four-footed shifts between the
    # pairs, the 40 mm it travels has the feet throw the body at 0.4 g and the
    # zero-moment point leaves the four feet; it moves through the swings as well.
    plan = []  # (seconds, y from, y to, steps, support, dip, bobs)
    stance, by, t = 0.0, lean[1], 0.0
    for i, d in enumerate(p["cycles"]):
        lead, trail = ("L", "R") if d > 0 else ("R", "L")
        mid = ready[1] + stance + d * p["stride"] / 2
        stance += d * p["stride"]
        square = ready.copy()
        square[1] = ready[1] + stance
        for j, pair in enumerate(((lead + "M", trail + "M"), (lead + "R", trail + "R"))):
            seconds = p["shift2_s"] if j else p["shift_s"]
            plan.append((seconds, by, mid, [], None, 0.0, 1))
            t += seconds
            by = mid
            for k, leg in enumerate(pair):
                lift = p["rear_lift"] if leg in rear else p["lift"]
                plan.append((p["swing_s"], by, by, [(leg, 0.0, 1.0, rig.place(square, leg), lift)],
                             tuple(f for f in four if f != leg), p["bob"], 1))
                t += p["swing_s"]
                if k == 0:
                    plan.append((p["double_s"], by, by, [],
                                 tuple(f for f in four if f != pair[1]), 0.0, 1))
                    t += p["double_s"]
        if i + 1 < len(p["cycles"]) and p["cycles"][i + 1] != d:
            plan.append((p["turn_s"], by, by, [], None, p["bob"], 2))
            t += p["turn_s"]
    # Squared up over the feet where they finally stand: the body is still midway
    # between them and where the last steps came from, and on the claws' way down
    # (`outro`) that leaves one middle leg folded 25 mm in under it.
    plan.append((p["square_s"], by, ready[1] + stance, [], None, 0.0, 1))
    t_end = t + p["square_s"]
    crossing = xy(four).mean(0)[0]

    # The centre of mass's fore-and-aft path, solved for rather than keyed: the
    # smoothest path (least acceleration, at the rows the builder will sample) that
    # holds it `margin` inside each triangle for the whole of every swing, starts and
    # ends where the four feet's diagonals cross, and is drawn gently toward that
    # crossing elsewhere. Keyed half cosines through the middle of each pair's lifts
    # (v9) left the triangle at the start and end of every swing unless clipped, and
    # a clip is a jump: past an 8 mm margin, the clip's jumps asked the feet for a
    # moment they cannot give in 8% of the rows (0.9 N*m at the 95th percentile).
    rows_t, lo, hi = [], [], []
    t = 0.0
    feet = {leg: m.feet[leg].copy() for leg in four}
    for seconds, y0, y1, steps, support, _dip, _bobs in plan:
        n = max(1, round(seconds / DT))
        for k in range(1, n + 1):
            u = k / n
            x = lean.copy()
            x[1] = y0 + (y1 - y0) * smooth(u)
            off = rig.com_offset(x, t + u * seconds)
            room = interval(support or four, x[1] + off[1], feet)
            if room is None:
                raise ValueError(f"t {t + u * seconds:.2f}: the centre of mass has no room "
                                 f"inside {support or four}")
            rows_t.append(t + u * seconds)
            lo.append(room[0])
            hi.append(room[1])
        for leg, _u0, _u1, target, _lift in steps:
            feet[leg] = target
        t += seconds
    rows_t = np.array(rows_t)
    n = len(rows_t) + 4  # two rows pinned to the crossing at each end
    pin = 1e-6
    lb = np.concatenate([[crossing - pin] * 2, lo, [crossing - pin] * 2])
    ub = np.concatenate([[crossing + pin] * 2, hi, [crossing + pin] * 2])
    accel = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)) / DT**2
    pull = sparse.identity(n) * p["centre_pull"]
    a = sparse.vstack([accel, pull]).tocsr()
    rhs = np.concatenate([np.zeros(n - 2), np.full(n, crossing * p["centre_pull"])])
    path = lsq_linear(a, rhs, bounds=(lb, ub), method="trf", lsmr_tol="auto",
                      max_iter=5000).x[2:-2]

    def com_x(tt: float) -> float:
        return float(np.interp(tt, rows_t, path, left=crossing, right=crossing))

    def fade(tt: float) -> float:
        return smooth(tt / 0.5) * smooth((t_end - tt) / 0.5)

    t = 0.0
    b = lean.copy()
    for seconds, y0, y1, steps, support, dip, bobs in plan:
        def base(u: float, t0=t, seconds=seconds, y0=y0, y1=y1, support=support, dip=dip,
                 bobs=bobs) -> np.ndarray:
            tt = t0 + u * seconds
            x = lean.copy()
            x[1] = y0 + (y1 - y0) * smooth(u)
            off = rig.com_offset(x, tt)
            cx = com_x(tt)
            if support is not None:
                # The claws' own swing moves the centre of mass sideways, and with it
                # where the diagonal is; held inside the triangle whatever they do.
                room = interval(support, x[1] + off[1])
                if room is None:
                    raise ValueError(f"t {tt:.2f}: the centre of mass has no room inside {support}")
                cx = float(np.clip(cx, *room))
            x[0] = cx - off[0]
            ph = 2 * np.pi * p["tremble_hz"] * tt
            x[2] = (p["crouch_z"] - dip * np.sin(bobs * np.pi * u) ** 2
                    + p["tremble_z"] * fade(tt) * np.sin(ph))
            return x

        m.segment(seconds, base, lambda u, t0=t, seconds=seconds: rig.dance(t0 + u * seconds),
                  steps, free=claws)
        b = base(1.0)
        t += seconds
    rig.outro(b, t, lean)
    return m


#: `shake_tripod` is the bear shake before both claws went up (gesture_shake v7, which
#: trained) and `shake_trot` the first with both up (v8, which trained to a shuffle):
#: kept to compare against while the crawl is new.
MOVES = {
    "flash": (flash, FLASH),
    "shake": (shake_crawl, SHAKE_CRAWL),
    "shake_trot": (shake_trot, SHAKE_TROT),
    "shake_tripod": (shake, SHAKE),
}


def synth(path: Path) -> Path:
    raw = path.read_bytes()
    spec = json.loads(raw)
    move, defaults = MOVES[spec["move"]]
    unknown = set(spec.get("params", {})) - set(defaults)
    if unknown:
        raise ValueError(f"{path.name}: {sorted(unknown)} are not {spec['move']}'s parameters")
    m = move({**defaults, **spec.get("params", {})})

    q, pose = np.array(m.rows_q), np.array(m.rows_b)
    lo, hi = m.robot.range[:, 0], m.robot.range[:, 1]
    over = np.maximum(lo - q.min(axis=0), q.max(axis=0) - hi)
    if (over > 1e-6).any():
        raise ValueError(f"{path.name}: outside the joint ranges: "
                         + ", ".join(f"{m.names[i]} by {over[i]:.3f} rad"
                                     for i in np.nonzero(over > 1e-6)[0]))
    n = len(q)
    f32 = np.float32
    out: dict[str, np.ndarray] = {
        "time": (np.arange(n) * DT).astype(f32),
        "dt": np.array(DT),
        "phase": np.ones(n, dtype=f32),
    }
    for prefix in ("", "meas_"):
        for i, a in enumerate(dances.AXES):
            out[f"{prefix}body_{a}"] = pose[:, i].astype(f32)
        for k, key in enumerate(dances._keys()):
            out[f"{prefix}{key}"] = q[:, k].astype(f32)
    out["source"] = np.array(f"steps:{path.name}")
    out["source_sha256"] = np.array(hashlib.sha256(raw).hexdigest())
    out["description"] = np.array(spec.get("description", ""))
    # The move starts and ends standing at HOME; there is no lead to add.
    out["lead_in_s"] = np.array(0.0)
    out["lead_out_s"] = np.array(0.0)
    out["hold_s"] = np.array(0.0)

    dest = path.with_name(f"{spec['move']}.npz")
    np.savez_compressed(dest, **out)

    speed = np.abs(np.diff(q, axis=0)).max(axis=0) / DT
    fastest = int(speed.argmax())
    travel = np.ptp(pose[:, :2], axis=0)
    print(f"{spec['move']}: {path.name} -> {dest}")
    print(f"  {n} rows at {1.0 / DT:.0f} Hz = {n * DT:.2f} s; leg IK misses by at most "
          f"{m.worst * 1000:.1f} mm, {len(m.missed)} row-legs over 3 mm")
    print(f"  base z {pose[:, 2].min() * 1000:.0f}..{pose[:, 2].max() * 1000:.0f} mm, "
          f"tilt at most {np.degrees(np.abs(pose[:, 3:5]).max()):.1f} deg, "
          f"twist {np.degrees(np.ptp(pose[:, 5])):.1f} deg, "
          f"travels {travel[0] * 1000:.0f} mm along x and {travel[1] * 1000:.0f} along y")
    print(f"  the centre of mass stays at least {m.margin[0] * 1000:.1f} mm inside the feet on the "
          f"floor (row {m.margin[1]}, {m.margin[1] * DT:.2f} s)")
    print(f"  fastest joint {m.names[fastest]} at {speed[fastest]:.2f} rad/s; a claw in the air "
          f"would have gone {max(0.0, -m.lowest) * 1000:.1f} mm through the floor")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("steps", type=Path, help="a <name>.steps.json file")
    args = ap.parse_args(argv)
    synth(args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
