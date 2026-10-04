"""jumper.gesture_cheer -- both claws raised overhead in a cheer, learned by imitation.

A one-shot gesture on the dance machinery, like `jumper.gesture_hello`: the
environment is `tasks/jumper/common/dance/env.py`, and what this directory adds is
the clip and the numbers.

## The clip

`media/cheer.npz`, written by `tools/synth_gesture.py` from the hand-placed
keyframes in `media/cheer.keyframes.json`. Nothing was recorded: this is the path
rl-wbc-fsm took for paw, and the first gesture in this repository made that way.

The robot leans back onto its four legs (the legs take hello's leaning stance),
raises both claws overhead, snaps them open and shut twice, sways them side to
side twice, snaps them once more and lowers them. The keyframes lean back
*before* the claws leave the floor and lower the claws *before* standing up
again: with both claws up the centre of mass is only 30 mm behind the middle
feet, and moving legs and arms together in the lead-in put it in front of them,
which `solve_resting_pose` answered with a robot tipped 63 degrees onto its
claws. Staged, the tilt stays within hello's 7.9 degrees.

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_cheer",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed cheer (both claws raised, "
                "snapped and swayed); the clip is committed in "
                "tasks/jumper/gesture_cheer/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
