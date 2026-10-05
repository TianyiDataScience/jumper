"""jumper.gesture_shake -- the bear shake, learned by imitation.

The crab's version of 狗熊哆嗦毛 ("the bear shakes its fur"), a move from Shandong
drum yangge that went viral in September 2026. A one-shot gesture on the dance
machinery, like `jumper.gesture_cheer`; `media/shake.npz` is written from
`media/shake.steps.json` by `tools/synth_steps.py`, because the move travels and
`tools/synth_gesture.py` can only write poses the robot holds at rest.

The clip follows the dancer in the reference video the user chose (a 2026 meme
compilation, measured frame by frame): six seconds without a pause in a deep horse
stance, a new stance every 0.25-0.5 s in a cross step, travelling across the stage
and back; the arms the biggest thing in it, swung through up overhead, crossed in
front and flung out flat, leading the torso's twist; the shoulders and the whole
body shuddering at 4-6 Hz; and a final pose with both arms raised.

The crab cannot swing both claws and step at once -- with both claws up its centre
of mass is over the middle feet, and no three legs hold it -- so it steps in
tripods, one always standing, and its claws take turns as the arms. Every beat one
tripod swings: its claw is flung into the next shape (up in a V, across in front,
out flat) while its two legs step, and the body bobs, twists with the claw and
leans into it. Seven beats travel 210 mm to the left, seven quicker ones back; the
body crouches to 94 mm (HOME stands at 107) and shudders at 5 Hz throughout,
harder towards the end; then it squares up, leans back over its four legs and
raises both claws, held 0.7 s.

Earlier clips showed that the momentum the claws carry decides what trains. The
first two whipped extended claws the same way as the body's yaw, 1518 and 688
mN*m of yaw torque at the 95th percentile for the feet to react, and neither
trained on an RTX 4070 (2026-10-05). The fourth swung guarded claws against the
twist, 29 mN*m, and trained (joint error 0.083) -- and was rejected by the user as
a shiver in place, not the dance: it had been designed from the move's name, before
the reference was at hand. This one asks 621, at the pace it does because faster
asked 808; `synth_steps.py` and `env_cfg.py` have the measurements.

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_shake",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the bear shake (a crouched tripod cross-step travelling sideways and back, the claws flung in turn, the body shuddering); the clip is committed in "
                "tasks/jumper/gesture_shake/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
