| iteration | Mean reward | Mean episode length | Metrics/motion/error_joint_pos | Episode_Termination/anchor_ori | Episode_Termination/ee_body_pos | Episode_Termination/time_out | Iteration time |
|---|---|---|---|---|---|---|---|
| 300 | 22.50 | 326.64 | 1.4459 | 0.0417 | 4.5417 | 8.4167 | 1.36s |
| 600 | 32.14 | 354.13 | 1.1970 | 0.0000 | 4.8750 | 8.2500 | 1.35s |
| 1000 | 33.79 | 358.64 | 1.0862 | 0.0000 | 3.1667 | 8.2500 | 1.36s |
| 2000 | 32.20 | 347.97 | 1.1853 | 0.0000 | 3.6250 | 7.7917 | 1.36s |
| 3000 | 34.21 | 370.19 | 1.2117 | 0.0000 | 4.3333 | 6.5000 | 1.36s |
| 3999 | 33.46 | 362.97 | 1.1257 | 0.0000 | 4.0833 | 7.4583 | 1.37s |

Trained on HEAD 7d8fdac Add the shake v10 (crawl, 18 mm margin) policy trained on an RTX 4070 (results only) (87 mm crouch clip) with the motion_support_feet_pos weight patched to 1.0 locally (not committed); run logs\jumper\jumper.gesture_shake\2026-10-06_09-23-08.
