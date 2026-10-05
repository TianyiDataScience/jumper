"""Hyper-parameters for jumper.gesture_shake.

jumper.dance's, value for value, and for the reasons written out in
`tasks/jumper/dance/rl_cfg.py`: the network, PPO and runner settings were chosen for
imitating a clip on this robot, not for that one clip. Written out again rather
than imported, so that changing one clip's training is a change to that clip.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config: `logs/<model>/jumper.gesture_shake/<date-time>`."""
    return jumper_ppo_baseline(
        experiment_name="jumper.gesture_shake",
        # Off, and not a tuning choice: the claws whip to the same side, both
        # shoulders turning by the same angle, which is not the mirror image of
        # itself, so mirroring its samples would train a different shake.
        # `symmetry.py` would also refuse the reference terms.
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # Half cheer's, and the entropy bonus a fifth of it. On the RTX 4070 the
        # shake stalled with cheer's 1.0 / 0.005: mean action std stayed at 0.20-0.21
        # at iteration 600 (cheer 0.04, flash 0.06) and 56-57% of episodes ended on
        # a support foot. The same model_600 played without the noise (64 envs, 500
        # steps) tripped nothing: LM and RR peaked at 49 mm against the 50 mm limit,
        # so the noise alone was carrying them over.
        init_std=0.5,
        entropy_coef=0.001,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=24,
        max_iterations=10_000,
    )


def runner_cls() -> type:
    """mjlab's motion-tracking runner, as jumper.dance: it logs the tracking metrics
    and the adaptive sampler's entropy, and bakes the clip into the exported ONNX."""
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    return MotionTrackingOnPolicyRunner
