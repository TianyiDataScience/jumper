"""Environment config for jumper.gesture_shake: the bear shake, `media/shake.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every
number. Where a number follows a rule `jumper.dance` wrote down for its own clip,
the rule is applied to this clip's measurements and the result is quoted -- read
`tasks/jumper/dance/env_cfg.py` for the reasoning behind each rule.

## This clip, measured

At the 50 Hz control rate (10.9 s): per-joint position std 0.261 rad at the median
and 0.682 at the most active, RMS amplitude 0.324 rad; joint velocity RMS 1.96
rad/s, per-joint std 1.69 at the median and 3.40 at the most active, peaking at
17.0 rad/s (RM_J2, a middle leg lifting 30 mm in a 0.22 s swing); the base
between 85 and 107 mm; no tilt. Official dances this machinery trains run at
1.5-2.8 rad/s RMS.

**The torque the feet must supply**, from the clip's angular momentum about its
centre of mass (kinematic, contacts off), at the 95th percentile: 225 mN*m in roll,
99 in pitch and 237 in yaw -- nearly all of it the claws'. The tripod clip before
it, which trained, asked 727, 354 and 619. The two shake clips that did not train
asked 1518 and 688 in yaw. A version that stepped the four legs one at a time and
brought a claw down to the floor for each middle leg asked 1027 / 660 / 1153: a
claw is 0.35 kg, 14% of the robot, and swung from overhead to the floor in an
eighth of a second it lifted the rear feet off the floor.

**Nothing but a policy holds this clip up.** For most of it the robot stands on
the line between two diagonal feet, with the centre of mass over that line; played
open loop on this task's actuators (PD to the reference, no policy) it survives
the claws going up and falls 0.2 s into the first swing. Two things had to be right
before that: the claws rise over a second, not 0.35 s, and the body is centred
between the middle and rear feet while they do -- on four legs at kp 10 a body
loaded onto the middle pair sags 20 mm in front and pitches onto its face, which it
did in every earlier version within a second of the claws leaving the floor.

**The reward weights are the tripod clip's**, which trained in 3000 iterations on
an RTX 4070 (2026-10-05) to joint error 0.200 and crossed 198 mm in steps lifted
34-39 mm; the stds follow this clip's measurements by jumper.dance's rules.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

#: This task's material, written by `tools/synth_steps.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. Half the 10.9 s clip, jumper.dance_maze's rule for a
#: short clip: the command teleports the robot back onto the reference wherever an
#: episode runs off the end, and at half the clip an episode sampled in its first
#: half runs clean.
EPISODE_S = 5.4


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.dance`'s for the arguments."""
    return dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        # ── Reference-state initialisation ────────────────────────────────
        # jumper.dance's: fractions of this robot (10 mm is a tenth of its standing
        # height), not of the clip, so they carry over unchanged.
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        rsi_joint_range=(-0.05, 0.05),
        # ── Observation noise ─────────────────────────────────────────────
        # The tracking skeleton's, as jumper.dance has them; not measured.
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # ── Rewards ───────────────────────────────────────────────────────
        # Positive budget 6.5 a step, as jumper.dance: mjlab's six tracking terms
        # unchanged (5.0) plus the two joint-space terms below (1.5).
        #
        # Joint position: jumper.dance's rule is a std at 0.44 of the clip's RMS
        # amplitude. Here: 0.44 x 0.324 -> 0.14.
        joint_pos_std=0.14,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS. Here: 1.96 -> 1.3.
        # Half the position weight, so when they disagree the pose wins.
        joint_vel_std=1.3,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        # Power: 22 joints at ~0.25 N*m and this clip's ~2 rad/s is ~11 W, so
        # -0.02 is about -0.22 a step, 3% of the budget -- jumper.dance's weight.
        power_weight=-0.02,
        # Torque above the continuous rating: anchored to the servo, not the clip
        # -- one joint at the plateau costs 0.6, 9% of the budget.
        torque_headroom_weight=-2.0,
        # Joint limits. The reference stays inside the soft joint limits throughout,
        # so this term charges a faithful policy nothing.
        joint_limit_weight=-10.0,
        # Leg-on-leg contact above 1.0 N, ~8% of the robot's worst-case push.
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations ──────────────────────────────────────────────────
        # Root height: this clip moves the body through 85-107 mm, a 22 mm band,
        # so 40 mm exceeds all of it and cannot fire on a policy merely tracking
        # badly -- jumper.dance's rule, and its number.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over"; the clip stays level.
        anchor_tilt_error=0.3,
        # A support foot 50 mm out vertically is half the standing height; this clip
        # lifts its stepping feet 27-30 mm. 70 mm was tried on an earlier clip and
        # did not save it; the clip was the problem.
        support_foot_error=0.05,
        # ── Disturbance ───────────────────────────────────────────────────
        # jumper.dance's: gentler and rarer than mjlab's, sized to a 2 kg robot.
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )
