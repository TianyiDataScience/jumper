"""jumper.gesture_shake -- the bear shake, learned by imitation.

The crab's version of 狗熊哆嗦毛 ("the bear shakes its fur"), a move from Shandong
drum yangge that went viral in September 2026. A one-shot gesture on the dance
machinery, like `jumper.gesture_cheer`; `media/shake.npz` is written from
`media/shake.steps.json` by `tools/synth_steps.py`, because the move travels and
`tools/synth_gesture.py` can only write poses the robot holds at rest.

The clip follows the dancer in the reference video the user chose (a 2026 meme
compilation, measured frame by frame): six seconds without a pause in a deep horse
stance, a new stance every 0.25-0.5 s in a cross step, travelling across the stage
and back; the arms the biggest thing in it and never below the waist -- a new
shape every quarter second, mostly mirror images (up in a V, crossed in front, out
flat) and every second or so both swung to the same side, the torso leaning after
them; the whole body shuddering at 4-6 Hz; and a final pose with both arms raised.

So the crab dances with both claws up from the first beat to the last, on its four
back legs. It drops into a wide, crouched stance (97 mm; HOME stands at 107),
settles back over the four legs, raises both claws over a second, and trots
sideways in diagonal pairs -- left middle with right rear, then right middle with
left rear -- a pair in the air for 0.22 s of every 0.28, each foot lifted 27-30 mm:
180 mm to the left, a beat on four feet as it turns, and back. On a pair's line the
crab is balanced the way a trotting dog is, the body moved fore and aft every beat
to keep the centre of mass over the line. The claws go through a new shape every
half second -- up in a V, crossed in front, out flat, both swung to one side --
shuddering against each other at 5 Hz while the body bobs and shudders with them;
it ends with both claws up in a W, held 0.7 s.

Earlier clips showed that the momentum the claws carry decides what trains. The
first two whipped extended claws the same way as the body's yaw, 1518 and 688
mN*m of yaw torque at the 95th percentile for the feet to react, and neither
trained on an RTX 4070 (2026-10-05). The fourth swung guarded claws against the
twist, 29 mN*m, and trained (joint error 0.083) -- and was rejected by the user as
a shiver in place, not the dance: it had been designed from the move's name, before
the reference was at hand. The sixth and seventh, built from the reference,
stepped in tripods with the claws taking turns as the arms; the seventh trained
(joint error 0.200) and crossed 198 mm in steps lifted 34-39 mm, but seen from the
front its claws were on the floor most of the time, where the dancer's arms never
are. A walk on the four legs one at a time, a claw coming down for each middle
leg, threw the claws too hard to stand. `synth_steps.py` and `env_cfg.py` have the
measurements.

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_shake",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the bear shake (both claws up and dancing while the four back legs trot sideways and back in diagonal pairs, crouched, the body shuddering); the clip is committed in "
                "tasks/jumper/gesture_shake/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
