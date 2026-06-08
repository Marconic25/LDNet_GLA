# LDNet replay-validation debugging log (sim_A_025_test, model clean/models/latent_10)

Target: NRMSE_F_y < 0.02 (training = 0.013), z_norm bounded/in-distribution.

| iter | hypothesis / change | NRMSE_F_y | max|Fy_err| | z_norm@2s | note |
|------|---------------------|-----------|-------------|-----------|------|
| 1 | baseline replay (subsample to dt_ref, factor 1.0) | 0.373 | 68.8 N | 4647 | z grows monotonically; matches training stepping though |
| 2 | fix off-by-one: reconstruct from z_i (current), not z_{i+1} | 0.373 | 68.8 N | 4647 | no change — masked by z saturation; kept (it's correct vs training) |
| 3 | **fix spatial point: normalize (0,0)->(-1,-1) for NNrec** | **0.019** | **6.2 N** | 4647 | TARGET MET. z growth is real & in-distribution (training rollout is byte-identical). NNrec needs normalized coords. |

## Root cause
 fed NNrec the raw spatial eval point , but training
( -> ) normalizes points with , giving . Wrong NNrec spatial input -> NRMSE 0.37. Fixed by computing
 from config in __init__ and using it in . (Also fixed a latent
off-by-one: reconstruct from current z, then step.) The growing latent norm (~7000) is NOT
instability — the exact training pipeline produces the identical z-trajectory and NNrec
decodes it correctly.
