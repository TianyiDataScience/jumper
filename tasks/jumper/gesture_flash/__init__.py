"""jumper.gesture_flash -- the flash step, learned by imitation.

The crab's version of 闪身步 ("the flash step"), a move from Anhui flower-drum
lantern dance that went viral in September 2026: a tilt and a sideways shift so
quick the dancer seems to teleport, then a freeze. A one-shot gesture on the dance
machinery, like `jumper.gesture_cheer`, and written the same way: `media/flash.npz`
comes from the keyframes in `media/flash.keyframes.json` through
`tools/synth_gesture.py`.

The robot takes cheer's leaning stance on its four legs with both claws raised on
guard, crouches 10 mm, and then snaps its body sideways -- 20 mm across, leaned 6
degrees into the move and twisted 11 degrees -- and freezes: left, right, left,
right, half a second held each time, before standing up. The feet do not move:
this machinery imitates a clip whose feet stay planted, so the flash is the body
over still feet, not a step. The body poses were solved with the four feet held
where they stand (a scratch least-squares IK on `_Robot`'s kinematics, within 1 mm
of every planted foot).

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_flash",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed flash step (the body snapped sideways over planted feet, left and right); the clip is committed in "
                "tasks/jumper/gesture_flash/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
