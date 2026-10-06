# jumper.gesture_shake gesture_shake_v12_beat105: shake on the dancer's beat (104.1 bpm)

model_3999.pt from logs\jumper\jumper.gesture_shake\2026-10-06_15-37-05, trained on e293acd Set the bear shake to the measured 104.1 bpm. Clip 26.26 s (1313 rows at 50 Hz). Measured with eval_v12.py: native CPU, play env, one uninterrupted pass from the clip's first row; nominal = every startup randomisation popped (base_com, encoder_bias, foot_friction), seed 1/2 = base_com and encoder_bias popped, foot friction drawn with the seed. v11 (model_3999, 2026-10-06_11-03-29) measured the same way on its own clip (18.86 s).

**Gate: PASS**

- PASS: no falls
- PASS: joint error <= v11 (+0.003 rad)
- PASS: lateral travel >= 90% of v11
- PASS: every leg lifts as often as v11's fewest
- PASS: lift heights >= 85% of v11
- PASS: claws never on a leg
- PASS: beat: mean |offset| after one lag <= 40 ms

## Against v11

| | v12 nominal | v12 seed 1 | v12 seed 2 | v11 nominal | v11 seed 1 | v11 seed 2 |
|---|---|---|---|---|---|---|
| seconds | 26.24 | 26.24 | 26.24 | 18.86 | 18.86 | 18.86 |
| falls / dones | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| peak tilt deg | 3.2 | 2.6 | 2.7 | 3.6 | 2.6 | 2.9 |
| joint error mean (p95) rad | 0.033 (0.061) | 0.033 (0.061) | 0.033 (0.061) | 0.037 (0.065) | 0.036 (0.065) | 0.036 (0.065) |
| travel x / y range mm (clip) | 118 / 157 (114 / 175) | 117 / 160 (114 / 175) | 114 / 158 (114 / 175) | 137 / 168 (115 / 175) | 132 / 167 (115 / 175) | 134 / 165 (115 / 175) |
| net travel cm | 1.8 | 2.9 | 1.8 | 4.2 | 4.0 | 3.4 |
| LM lift-offs >25 mm (clip), max mm | 8 (8), 41 | 8 (8), 37 | 8 (8), 38 | 7 (8), 45 | 7 (8), 40 | 7 (8), 42 |
| RM lift-offs >25 mm (clip), max mm | 8 (8), 40 | 8 (8), 37 | 8 (8), 37 | 4 (8), 42 | 4 (8), 37 | 4 (8), 38 |
| LR lift-offs >25 mm (clip), max mm | 8 (8), 41 | 8 (8), 37 | 8 (8), 38 | 8 (8), 45 | 8 (8), 41 | 8 (8), 42 |
| RR lift-offs >25 mm (clip), max mm | 8 (8), 41 | 8 (8), 37 | 8 (8), 38 | 8 (8), 44 | 8 (8), 41 | 8 (8), 42 |
| crouch mean / deepest % (clip) | 15 / 27 (18 / 30) | 14 / 25 (18 / 30) | 15 / 26 (18 / 30) | 20 / 33 (18 / 29) | 19 / 31 (18 / 29) | 19 / 32 (18 / 29) |
| shudder Hz, rms mm (clip) | 5.2, 2.1 (3.3) | 5.2, 2.1 (3.3) | 5.2, 2.1 (3.3) | 5.0, 2.0 (3.3) | 5.0, 2.0 (3.3) | 5.0, 2.0 (3.3) |
| claw tip height error mm | 8.8 | 6.6 | 7.5 | 8.0 | 6.0 | 6.8 |
| claw nearest a leg mm L/R (rows touching) | 10.9/10.5 (0) | 10.9/10.8 (0) | 10.9/10.6 (0) | 10.8/11.6 (0) | 10.8/11.6 (0) | 10.7/11.6 (0) |

## Beat table (nominal rollout)

Grid: beat k at k x 60/104.1 = k x 576.4 ms from row 0. Claw shape reached = the row where the clip's claw J2/J3 stop (speed 0); touchdown = the row where the clip's foot stops coming down. The robot's time for each = the clip's plus the shift (sub-row) that best lays the robot's claw J2/J3, or that foot's height, over the clip's through the approach. The direct reading (claws under 0.5 rad/s after their peak; foot within 3 mm of where it then stands) is a cross-check. Offsets in ms (+ = late).

- claw shapes: n=35 (missed 1), robot mean |offset| 14, max 35; systematic lag +13; after removing it mean 7, max 22. The clip itself: mean 4.9, max 9.3
- touchdowns: n=32 (missed 0), robot mean |offset| 9, max 25; systematic lag +8; after removing it mean 6, max 17. The clip itself: mean 5.1, max 9.3
- all events: n=67 (missed 1), robot mean |offset| 11, max 35; systematic lag +11; after removing it mean 7, max 24. The clip itself: mean 5.0, max 9.3
- all events, direct reading: n=68 (missed 0), robot mean |offset| 24, max 238; systematic lag +0; after removing it mean 24, max 238. The clip itself: mean 5.0, max 9.3

| event | beat | grid s | clip off ms | robot s | robot off ms |
|---|---|---|---|---|---|
| claw | 5 | 2.882 | -1.8 | 2.886 | +4 |
| claw | 6 | 3.458 | +1.8 | 3.459 | +1 |
| LM down | 6 | 3.458 | +1.8 | 3.459 | +0 |
| claw | 7 | 4.035 | +5.4 | 4.061 | +27 |
| RM down | 7 | 4.035 | +5.4 | 4.046 | +11 |
| claw | 8 | 4.611 | +9.0 | 4.638 | +27 |
| LR down | 8 | 4.611 | +9.0 | 4.636 | +25 |
| claw | 9 | 5.187 | -7.3 | 5.200 | +12 |
| RR down | 9 | 5.187 | -7.3 | 5.186 | -1 |
| claw | 10 | 5.764 | -3.7 | 5.769 | +5 |
| LM down | 10 | 5.764 | -3.7 | 5.773 | +9 |
| claw | 11 | 6.340 | -0.1 | 6.357 | +17 |
| RM down | 11 | 6.340 | -0.1 | 6.349 | +9 |
| claw | 12 | 6.916 | +3.6 | 6.926 | +10 |
| LR down | 12 | 6.916 | +3.6 | 6.934 | +17 |
| claw | 13 | 7.493 | +7.2 | 7.514 | +21 |
| RR down | 13 | 7.493 | +7.2 | 7.503 | +10 |
| claw | 14 | 8.069 | -9.2 | 8.069 | +0 |
| LM down | 14 | 8.069 | -9.2 | 8.064 | -6 |
| claw | 15 | 8.646 | -5.5 | 8.658 | +13 |
| RM down | 15 | 8.646 | -5.5 | 8.647 | +1 |
| claw | 16 | 9.222 | -1.9 | 9.233 | +11 |
| LR down | 16 | 9.222 | -1.9 | 9.229 | +8 |
| claw | 17 | 9.798 | +1.7 | 9.810 | +12 |
| RR down | 17 | 9.798 | +1.7 | 9.808 | +10 |
| claw | 18 | 10.375 | +5.4 | 10.390 | +15 |
| LM down | 18 | 10.375 | +5.4 | 10.385 | +10 |
| claw | 19 | 10.951 | +9.0 | 10.976 | +25 |
| RM down | 19 | 10.951 | +9.0 | 10.969 | +18 |
| claw | 20 | 11.527 | -7.4 | 11.528 | +0 |
| LR down | 20 | 11.527 | -7.4 | 11.536 | +9 |
| claw | 21 | 12.104 | -3.7 | 12.113 | +10 |
| RR down | 21 | 12.104 | -3.7 | 12.103 | -1 |
| claw | 22 | 12.680 | -0.1 | 12.693 | +13 |
| claw | 23 | 13.256 | +3.5 | 13.279 | +23 |
| RM down | 23 | 13.256 | +3.5 | 13.264 | +8 |
| claw | 24 | 13.833 | +7.1 | 13.855 | +22 |
| LM down | 24 | 13.833 | +7.1 | 13.853 | +20 |
| claw | 25 | 14.409 | -9.2 | 14.414 | +5 |
| RR down | 25 | 14.409 | -9.2 | 14.413 | +4 |
| claw | 26 | 14.986 | -5.6 | 14.993 | +7 |
| LR down | 26 | 14.986 | -5.6 | 14.987 | +1 |
| claw | 27 | 15.562 | -2.0 | 15.580 | +18 |
| RM down | 27 | 15.562 | -2.0 | 15.568 | +6 |
| claw | 28 | 16.138 | +1.7 | 16.150 | +12 |
| LM down | 28 | 16.138 | +1.7 | 16.139 | +1 |
| claw | 29 | 16.715 | +5.3 | 16.742 | +27 |
| RR down | 29 | 16.715 | +5.3 | 16.733 | +18 |
| claw | 30 | 17.291 | +8.9 | 17.318 | +27 |
| LR down | 30 | 17.291 | +8.9 | 17.303 | +12 |
| claw | 31 | 17.867 | -7.4 | 17.870 | +3 |
| RM down | 31 | 17.867 | -7.4 | 17.866 | -2 |
| claw | 32 | 18.444 | -3.8 | 18.456 | +12 |
| LM down | 32 | 18.444 | -3.8 | 18.456 | +12 |
| claw | 33 | 19.020 | -0.2 | 19.036 | +16 |
| RR down | 33 | 19.020 | -0.2 | 19.031 | +11 |
| claw | 34 | 19.597 | +3.5 | 19.609 | +12 |
| LR down | 34 | 19.597 | +3.5 | 19.601 | +5 |
| claw | 35 | 20.173 | +7.1 | 20.208 | +35 |
| RM down | 35 | 20.173 | +7.1 | 20.188 | +15 |
| claw | 36 | 20.749 | -9.3 | 20.749 | -0 |
| LM down | 36 | 20.749 | -9.3 | 20.740 | -9 |
| claw | 37 | 21.326 | -5.6 | 21.340 | +14 |
| RR down | 37 | 21.326 | -5.6 | 21.334 | +8 |
| claw | 38 | 21.902 | -2.0 | 21.915 | +13 |
| LR down | 38 | 21.902 | -2.0 | 21.904 | +2 |
| claw | 40 | 23.055 | +5.2 | 23.072 | +17 |
| claw | 45 | 25.937 | +3.4 | - | - |
