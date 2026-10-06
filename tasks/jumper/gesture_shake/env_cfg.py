"""Environment config for jumper.gesture_shake: the bear shake, `media/shake.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every
number. Where a number follows a rule `jumper.dance` wrote down for its own clip,
the rule is applied to this clip's measurements and the result is quoted -- read
`tasks/jumper/dance/env_cfg.py` for the reasoning behind each rule.

## This clip, measured

At the 50 Hz control rate (18.9 s): per-joint position std 0.309 rad at the median
and 0.867 at the most active (a claw's wrist), RMS amplitude 0.447 rad; joint
velocity RMS 2.24 rad/s, per-joint std 2.18 at the median and 3.67 at the most
active, peaking at 16.3 rad/s (LM_J2, a middle leg lifting 30 mm in a 0.22 s
swing); the base between 75 and 107 mm; no tilt. Official dances this machinery
trains run at 1.5-2.8 rad/s RMS.

**What the feet must supply**, from the clip's inverse dynamics (contacts off, the
support feet's forces solved for, pushing only): everything, outside a row or two
at a touchdown -- the moment they cannot supply is 0.000 N*m at the 95th
percentile, as for the tripod clip that trained (v7). The trot before this one
(v8) stood on two feet for 0.22 s of every 0.28 and asked 0.095 N*m that no foot
could give, in 14% of its rows; its policy, trained on an RTX 4070 to 3000
iterations (2026-10-06), kept the middle feet on the floor -- 17 mm lifts against
30 -- and crossed 9 cm of the 18. Here one leg is in the air at a time and the
centre of mass is inside the other three's triangle for the whole swing, at least
17.8 mm in, the body swaying fore and aft to put it there; the moment the feet
cannot supply is over 0.05 N*m in 2% of the rows, all at touchdowns. The support
legs hold 1.31-1.54 N*m at the 95th percentile (J1). The claws' angular momentum
asks 261 mN*m of roll, 214 of pitch and 258 of yaw (95th percentile, kinematic);
the tripod clip asked 727 of roll and 619 of yaw.

**v9 held it 8 mm in, and its middle feet never left the floor.** Its policy (RTX
4070, 4000 iterations, 2026-10-06) stepped the rear feet 31-35 mm and kept the
middle feet down -- 0 and 1 lift-offs of 25 mm, travel 13.4 cm -- with its centre
of mass 10-15 mm forward of the clip's, -7 to +5 mm from the edge of every middle
leg's triangle: robust to the base's randomised centre of mass (below), which on
the skeleton's ranges moved the robot's by as much as the clip's margin.

**v10 held it 18 mm in, and its left middle foot still never left the floor** (0
lift-offs, at most 11 mm, against the right one's 3 of 38-42 mm; RTX 4070, 4000
iterations, 2026-10-06, and again with the feet's weight at 1.0). The claws' shapes,
solved by their tips alone, held each palm upright from a wrist out at the middle
leg's knee: 20-50 mm through the middle legs in every shape but one, which the
simulator does not allow, so no policy can follow both claw and leg. v10b's (played
without randomisation) held its left claw within 5 mm of the left middle leg for 38%
of the dance, its claw yaws 0.39-0.40 rad off the clip's, and kept that leg down: the
clip's own leg, against the claw where the policy held it, would have gone up to 12 mm
into it in 31% of the rows. Each shape now turns its claw forward at the
shoulder to clear the legs by 9-11 mm in every way the crawl puts them, the tips 2-5
cm further forward (`tools/synth_steps.py`), and nowhere in the clip do two limbs
come nearer each other than 0.9 mm.

**Played open loop it still falls**: on this task's actuators with PD to the
reference and no policy, the body sags 15 mm onto the soft servos (kp 10) while
the claws go up and pitches over 2.7 s in -- as the trot did, whose policy then
danced it without a fall. The claws rise over a second, not 0.35 s, and the body
is centred between the middle and rear feet while they do: on four legs at kp 10
a body loaded onto the middle pair sags 20 mm in front and pitches onto its face.

**The reward weights are the tripod clip's**, which trained in 3000 iterations on
an RTX 4070 (2026-10-05) to joint error 0.200 and crossed 198 mm in steps lifted
34-39 mm; the stds follow this clip's measurements by jumper.dance's rules. One
term is added: the four support feet's own positions.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.tracking import mdp

from ..common.dance.env import COMMAND, SUPPORT_FEET, dance_env_cfg

#: This task's material, written by `tools/synth_steps.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. Half the 18.9 s clip, jumper.dance_maze's rule for a
#: short clip: the command teleports the robot back onto the reference wherever an
#: episode runs off the end, and at half the clip an episode sampled in its first
#: half runs clean.
EPISODE_S = 9.4


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.dance`'s for the arguments."""
    cfg = dance_env_cfg(
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
        # unchanged (5.0) plus the two joint-space terms below (1.5); and 0.5 for
        # the support feet, below the call.
        #
        # Joint position: jumper.dance's rule is a std at 0.44 of the clip's RMS
        # amplitude. Here: 0.44 x 0.447 -> 0.20.
        joint_pos_std=0.20,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS. Here: 2.24 -> 1.50.
        # Half the position weight, so when they disagree the pose wins.
        joint_vel_std=1.50,
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
        # Root height: this clip moves the body through 75-107 mm, a 32 mm band,
        # so 40 mm exceeds all of it and cannot fire on a policy merely tracking
        # badly -- jumper.dance's rule, and its number.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over"; the clip stays level.
        anchor_tilt_error=0.3,
        # A support foot 50 mm out vertically is half the standing height; this clip
        # lifts its stepping feet 30 mm. 70 mm was tried on an earlier clip and
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
    # The support feet where the clip puts them, the four of them averaged: mjlab's
    # body-position term scores them too, but at its 0.3 m std a foot that shuffles
    # 15 mm off the floor instead of stepping 30 mm costs it nothing, and that is
    # what the trot's policy did. At 0.03 m a foot left on the floor through its
    # swing costs a fifth of this term while it is down; half the weight of the
    # joint-position term, which already charges for it in joint space.
    cfg.rewards["motion_support_feet_pos"] = RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.5,
        params={"command_name": COMMAND, "std": 0.03, "body_names": SUPPORT_FEET},
    )
    # The base's centre-of-mass offset, randomised once per environment and invisible
    # to the policy. The tracking skeleton's +/-25/50/50 mm are the G1 torso's; on this
    # 0.886 kg base of a 2.54 kg robot they move the whole robot's centre of mass
    # +/-8.7 mm fore and aft and +/-17 mm sideways -- as much as the crawl's whole
    # margin inside its three feet. The v9 policy (RTX 4070, 4000 iterations,
    # 2026-10-06) learned the one stance that survives all of it: centre of mass
    # 10-15 mm forward of the clip's, -7 to +5 mm from the edge of every middle leg's
    # triangle, so the middle feet never left the floor (0 and 1 lift-offs of 25 mm
    # against the rear legs' 3-4). +/-10 mm on the base is +/-3.5 mm on the robot.
    # Still not measured on this robot -- nor were the skeleton's.
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01),
    }
    return cfg
