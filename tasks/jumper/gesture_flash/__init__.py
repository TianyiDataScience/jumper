"""jumper.gesture_flash -- the flash step, learned by imitation.

The crab's version of 闪身步 ("the flash step"), a move from Anhui flower-drum
lantern dance that went viral in September 2026. A one-shot gesture on the dance
machinery, like `jumper.gesture_cheer`; `media/flash.npz` is written from
`media/flash.steps.json` by `tools/synth_steps.py`, because the move steps and
`tools/synth_gesture.py` can only write poses the robot holds at rest.

The clip follows the dancer in the reference video the user chose (a 2026 meme
compilation; the flash step is its first 8 s, measured frame by frame): stillness,
then one sharp move repeated every 1.6-1.7 s. A 0.1 s sway, a 0.18 s flash -- one
foot shoots out sideways, the knees drop the hips 12-15%, the torso leans 20-25
degrees away over the planted leg and twists, the arm on the foot's side is flung
out low and wide -- held 0.12 s, snapped back in 0.18 s, the body a third of a
shoulder width over; still again, then the other way.

On the crab the middle leg on the flash side shoots out sideways, lifted 30 mm, to
land about 58 mm out; the body drops from 107 to 90 mm, leans 20.6 degrees over
the other side and twists 11.5 degrees against the claw on the flash side, which
is flung out low and wide, jaw open; then all of it snaps back, the four legs left
behind scuttle after it in two quick pairs, each foot lifted 25 mm, and the crab
stands still 80 mm over. Left, right, left, right, every 1.6 s. `synth_steps.py`
has why each number is where it is.

An earlier clip snapped the body 20 mm sideways over feet that never moved. It
trained (joint error 0.121 on an RTX 4070, 2026-10-05) and was rejected by the user
as a sway, not the dance: it had been designed from the move's name, before the
reference was at hand. The one before this one carried the body 80 mm and kept
the dancer's rhythm once trained (joint error 0.084), but it leaned 10 degrees
against the dancer's 20-25, and its feet slid out and after it at most 16 mm off
the floor, where no one could see them step.

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_flash",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the flash step (a middle leg shot out sideways, the body dropped and leaned away, a claw flung, 8 cm over; left and right); the clip is committed in "
                "tasks/jumper/gesture_flash/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
