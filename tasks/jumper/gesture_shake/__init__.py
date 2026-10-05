"""jumper.gesture_shake -- a bear shaking off water, learned by imitation.

The crab's version of 狗熊哆嗦毛 ("the bear shakes its fur"), a move from Shanghe
drum yangge that went viral in September 2026: knees bent, the waist drives the
whole body into a fast shiver. A one-shot gesture on the dance machinery, like
`jumper.gesture_cheer`, and written the same way: `media/shake.npz` comes from the
keyframes in `media/shake.keyframes.json` through `tools/synth_gesture.py`.

The robot takes cheer's leaning stance on its four legs, crouches 12 mm and holds
both claws up on guard (flash's arm pose), then rolls and twists its body from side
to side while the claws swing **against** the twist and snap open and shut -- two
slow shakes, three faster, four fast -- and freezes, claws shut, before standing up.
The body poses were solved with the four feet held where they stand (a scratch
least-squares IK on `_Robot`'s kinematics, within 1 mm of every planted foot), so
each keyframe is a pose the robot holds without sliding.

This is the third clip, and the first two are why the claws counter-swing. Both
whipped the claws the same way the body yawed, with the arms flung out sideways, so
the swing's angular momentum added to the body's and the four feet had to react
all of it: 1518 and 688 mN*m of yaw torque at the 95th percentile, against flash's
361. On an RTX 4070 (2026-10-05) neither trained -- a joint-position error of 0.65-0.68
at iteration 600-1500 (cheer 0.078, flash 0.121) with over half the episodes ending
on a middle leg kicked off the floor -- and neither lower exploration noise nor a
looser foot bound changed that. Counter-swinging guarded claws bring the demand to
29 mN*m. `env_cfg.py` has the measurements.

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_shake",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed bear shake (crouched, the body shaken side to side, the claws swinging against it); the clip is committed in "
                "tasks/jumper/gesture_shake/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
