"""Environment config for jumper.gesture_shake: the bear shaking off water, `media/shake.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every
number. Where a number follows a rule `jumper.dance` wrote down for its own clip,
the rule is applied to this clip's measurements and the result is quoted -- read
`tasks/jumper/dance/env_cfg.py` for the reasoning behind each rule.

## This clip, measured

At the 50 Hz control rate (10.0 s, lead-in and lead-out included): per-joint
position std 0.131 rad at the median and 0.366 at the most active, RMS amplitude
0.159 rad; joint velocity RMS 0.687 rad/s, per-joint std 0.591 at the median and
1.087 at the most active, peaking at 2.90 rad/s (LF_J4, the claws snapping); the
base between 103 and 115 mm; tilt at most 8.0 degrees.

**The yaw torque the feet must supply**, from the clip's angular momentum about its
centre of mass (kinematic, contacts off): 29 mN*m at the 95th percentile. The two
clips that did not train asked 1518 (first) and 688 (second), with the claws
whipping the same way as the body; flash, which trained, asks 361 and cheer 51.
Roll: 94 mN*m (flash 178).

Played open loop on this task's actuators (PD to the reference, no policy), the
clip runs all 498 steps without a termination: joint error median 0.013 rad, p95
0.104, worst 0.191 (LM_J1) -- cheer's figures, 0.011 / 0.087 / 0.226; the second
clip's were 0.026 / 0.147 / 0.517. A support foot is at most 16.6 mm off its
reference height. In the training environment (16 envs from the clip's start,
randomisation and pushes on, action = reference) no support-foot termination fires
and the joint-error norm averages 0.225 (flash 0.241, the second clip 0.369).

**The reward weights are starting points**: no policy has been trained on this
clip yet.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

#: This task's material, written by `tools/import_wbc_gestures.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. Half the 10.0 s clip, jumper.dance_maze's rule for a
#: short clip: the command teleports the robot back onto the reference wherever an
#: episode runs off the end, and at half the clip an episode sampled in its first
#: half runs clean.
EPISODE_S = 5.0


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
        # amplitude. Here: 0.44 x 0.159 -> 0.07.
        joint_pos_std=0.07,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS. Here: 0.687 -> 0.46.
        # Half the position weight, so when they disagree the pose wins.
        joint_vel_std=0.46,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        # Power: 22 joints at ~0.25 N*m and this clip's ~0.69 rad/s is ~3.8 W, so
        # -0.02 is about -0.08 a step, 1% of the budget -- jumper.dance's weight.
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
        # Root height: this clip moves the body through 103-115 mm, a 12 mm band,
        # so 40 mm exceeds all of it and cannot fire on a policy merely tracking
        # badly -- jumper.dance's rule, and its number.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over"; the clip leans 8.0.
        anchor_tilt_error=0.3,
        # A support foot 50 mm out vertically is half the standing height. 70 mm was
        # tried on the second clip and did not save it; the clip was the problem.
        support_foot_error=0.05,
        # ── Disturbance ───────────────────────────────────────────────────
        # jumper.dance's: gentler and rarer than mjlab's, sized to a 2 kg robot.
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )
