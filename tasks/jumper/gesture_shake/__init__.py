"""jumper.gesture_shake -- a bear shaking off water, learned by imitation.

The crab's version of 狗熊哆嗦毛 ("the bear shakes its fur"), a move from Shanghe
drum yangge that went viral in September 2026: knees bent, the waist drives the
whole body into a fast shiver. A one-shot gesture on the dance machinery, like
`jumper.gesture_cheer`, and written the same way: `media/shake.npz` comes from the
keyframes in `media/shake.keyframes.json` through `tools/synth_gesture.py`.

The robot takes cheer's leaning stance on its four legs, crouches 12 mm and flings
both claws out sideways at shoulder height, then rolls and twists its body from
side to side while the claws whip the other way -- two slow shakes, three faster,
five fast -- and freezes, claws shut, before standing up. The body poses were
solved with the four feet held where they stand (a scratch least-squares IK on
`_Robot`'s kinematics, within 1 mm of every planted foot), so each keyframe is a
pose the robot holds without sliding. The fast shake is the clip's limit: 0.26 s
a cycle, smaller in amplitude, so no joint passes 5.2 rad/s.

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_shake",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed bear shake (crouched, claws flung out, the body shaken side to side); the clip is committed in "
                "tasks/jumper/gesture_shake/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
