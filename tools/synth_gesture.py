#!/usr/bin/env python3
"""Write a gesture clip from hand-placed keyframes, for a task to learn by imitation.

    python tools/synth_gesture.py tasks/jumper/gesture_cheer/media/cheer.keyframes.json

The four imported gestures (`import_wbc_gestures.py`) came from recordings, except
paw, which rl-wbc-fsm interpolated from hand-placed keyframes. This is that second
path, for a gesture nobody has recorded: a short JSON file of keyframes in, the
`.npz` the gesture tasks read out, next to it.

## The keyframe file

    {
      "name": "cheer",
      "description": "one line, kept in the clip",
      "dt": 0.02,
      "keyframes": [
        {"t": 0.0, "set": {"LF_J1_joint": -2.6, "RF_J1_joint": -2.6}},
        {"t": 0.5, "set": {"LF_J4_joint": -0.8}},
        {"t": 0.8}
      ]
    }

Each keyframe starts from the one before it -- the first from `HOME` -- and
changes only the joints its `set` names, with the robot's own joint names. A
keyframe with no `set` holds the pose. Between keyframes every joint moves on a
smoothstep, so each keyframe is a pose the robot passes through at rest. `t` is
seconds from the first keyframe and must increase.

## What is done to it, and why it is the importer's

Everything after the keyframes is `import_wbc_gestures.py`'s, called rather than
restated: values are clamped to the joint ranges (and each clamp is reported), a
lead-in from `HOME` and a lead-out back to it are added at `LEAD_SPEED`, the clip
ends with `HOLD_S` standing at `HOME`, and the base pose is solved from all six
feet with `solve_resting_pose`. That last step is also the clip's first check:
a row whose centre of mass no three feet can hold up raises there, and a pose
that would topple is refused before any training sees it.

It does **not** check that physics can play the clip. A pose can rest on the
floor and still be one the servos cannot reach in time; that is what training it
finds out, and what a playback of the clip on the task's own actuators measures
before training (a one-off probe, not part of this tool).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# `tools/` is not a package; run as a script, its own directory is on `sys.path`.
import import_wbc_dances as dances
import import_wbc_gestures as gestures
import numpy as np


def keyframe_rows(spec: dict, names: list[str], home: np.ndarray) -> np.ndarray:
    """The keyframes, interpolated at `spec["dt"]`: [T, 22] in `names` order."""
    dt = float(spec["dt"])
    frames = spec["keyframes"]
    if not frames:
        raise ValueError("no keyframes")
    poses, times = [], []
    pose = home.copy()
    for k, frame in enumerate(frames):
        pose = pose.copy()
        for joint, value in frame.get("set", {}).items():
            if joint not in names:
                raise ValueError(f"keyframe {k}: {joint!r} is not one of the robot's joints")
            pose[names.index(joint)] = float(value)
        t = float(frame["t"])
        if times and t <= times[-1]:
            raise ValueError(f"keyframe {k}: t={t} does not come after t={times[-1]}")
        poses.append(pose)
        times.append(t)
    rows = [poses[0][None]]
    for (a, ta), (b, tb) in zip(zip(poses, times), zip(poses[1:], times[1:])):
        n = max(1, round((tb - ta) / dt))
        u = gestures._smoothstep(np.arange(1, n + 1) / n)[:, None]
        rows.append(a + (b - a) * u)
    return np.concatenate(rows)


def synth(path: Path) -> Path:
    from tasks.jumper.common.constants import HOME

    raw = path.read_bytes()
    spec = json.loads(raw)
    name = spec["name"]
    dt = float(spec["dt"])
    robot = gestures._Robot()
    home = np.array([HOME[n] for n in robot.names])

    q = keyframe_rows(spec, robot.names, home)
    lo, hi = robot.range[:, 0], robot.range[:, 1]
    over = np.maximum(lo - q.min(axis=0), q.max(axis=0) - hi)
    clamped = [(robot.names[i], over[i]) for i in np.nonzero(over > 0.0)[0]]
    q = np.clip(q, lo, hi)

    lead_in, t_in = gestures._lead(home, q[0], dt)
    lead_out, t_out = gestures._lead(q[-1], home, dt)
    hold = np.repeat(home[None], round(gestures.HOLD_S / dt) + 1, axis=0)
    q = np.concatenate([lead_in, q, lead_out, hold])

    pose, height = gestures.solve_resting_pose(robot, q)
    lifted = {f: float((height[:, i] >= gestures.DOWN_TOL).mean())
              for i, f in enumerate(gestures.FEET)}

    n = len(q)
    f32 = np.float32
    out: dict[str, np.ndarray] = {
        "time": (np.arange(n) * dt).astype(f32),
        "dt": np.array(dt),
        "phase": np.ones(n, dtype=f32),
    }
    # Commanded and measured are the same track: nothing was measured, and the
    # gesture tasks read `command` (see `import_wbc_dances.py` for that choice).
    for prefix in ("", "meas_"):
        for i, a in enumerate(dances.AXES):
            out[f"{prefix}body_{a}"] = pose[:, i].astype(f32)
        for k, key in enumerate(dances._keys()):
            out[f"{prefix}{key}"] = q[:, k].astype(f32)
    out["source"] = np.array(f"keyframes:{path.name}")
    out["source_sha256"] = np.array(hashlib.sha256(raw).hexdigest())
    out["description"] = np.array(spec.get("description", ""))
    out["lead_in_s"] = np.array(t_in)
    out["lead_out_s"] = np.array(t_out)
    out["hold_s"] = np.array(gestures.HOLD_S)

    dest = path.with_name(f"{name}.npz")
    np.savez_compressed(dest, **out)

    body = n - len(lead_in) - len(lead_out) - len(hold)
    speed = np.abs(np.diff(q, axis=0)).max(axis=0) / dt
    fastest = int(speed.argmax())
    print(f"{name}: {path.name} -> {dest}")
    print(f"  {body} rows at {1.0 / dt:.0f} Hz = {body * dt:.2f} s, plus {t_in:.2f} s in, "
          f"{t_out:.2f} s out, {gestures.HOLD_S:.2f} s held = {n * dt:.2f} s")
    print(f"  base z {pose[:, 2].min() * 1000:.0f}..{pose[:, 2].max() * 1000:.0f} mm, "
          f"tilt at most {np.degrees(np.abs(pose[:, 3:5]).max()):.1f} deg, "
          f"travels {np.hypot(*np.ptp(pose[:, :2], axis=0)) * 1000:.0f} mm")
    print(f"  fastest joint {robot.names[fastest]} at {speed[fastest]:.2f} rad/s")
    print("  off the floor: " + ", ".join(f"{f} {v * 100:.0f}%" for f, v in lifted.items()))
    for jname, by in clamped:
        print(f"  clamped {jname} by up to {by * 1000:.1f} mrad")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("keyframes", type=Path, help="a <name>.keyframes.json file")
    args = ap.parse_args(argv)
    synth(args.keyframes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
