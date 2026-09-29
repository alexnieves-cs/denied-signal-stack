# ovserver protocol (v1)

`ovserver` runs OpenVINS (GPL-3.0) in its own process. The MIT Python package talks to it over
stdin/stdout, one strict request → one response, in lockstep. At startup, fd 1 is `dup`ed for the
protocol and then pointed at stderr, so OpenVINS's `printf`s never corrupt the stream.

## Framing
Every message in either direction is framed as:

```
uint32 payload_len   (little-endian, bytes after this 5-byte header; max 64 MiB)
uint8  type
payload[payload_len]
```

All numbers are little-endian: `f64` is IEEE double, `f32` is float, `uN` is unsigned.
If a request claims more than 64 MiB, the server reads and discards those bytes, then answers `ACK(BAD_REQUEST)`.
EOF on stdin makes the server exit 0. A request the server can't parse gets an error status and the server stays up.

## Requests
| type | name | payload |
|---|---|---|
| 1 | `INIT` | UTF-8 path to an OpenVINS `estimator_config.yaml` (its `relative_config_*` files sit beside it). Creates a fresh estimator. Multi-threading is forced off (lockstep). |
| 2 | `IMU` | `f64 t, wx, wy, wz, ax, ay, az` (56 bytes; s, rad/s, m/s²). `t` must strictly increase. |
| 3 | `CAM` | `f64 t, u8 cam_id, u32 width, u32 height, u8 gray[width*height]` (row-major CV_8UC1) |
| 7 | `CAM_STEREO` | `f64 t`, then two `(u8 cam_id, u32 w, u32 h, u8 gray[w*h])` blocks with different ids |
| 4 | `INIT_GT` | 17 × `f64`: `t, q_GtoI (JPL x,y,z,w), p_IinG[3], v_IinG[3], bg[3], ba[3]`. Calls `initialize_with_gt` (OpenVINS's hard-coded static covariance) and raises the tracker to `num_pts`. |
| 5 | `REANCHOR` | the 17 `INIT_GT` values, then 225 × `f64`: a 15×15 row-major covariance in OpenVINS IMU error order **[θ, p, v, bg, ba]** (JPL θ). Builds a **fresh** estimator (no clones, no features), replays the last 3 s of IMU, and seeds it with this covariance. Rejected unless it is positive definite. |
| 6 | `SHUTDOWN` | empty; the server ACKs and exits 0 |
| 8 | `PING` | empty; ACK with text `ovserver 1` |

**Ordering rule:** send every IMU sample up to and past the camera time before its `CAM`
(`last_imu_t ≥ t_cam + calib_camimu_dt`). Otherwise the frame is not processed and the reply
status is `NEED_IMU`. OpenVINS propagates to the camera time and needs IMU data there.

Validation happens before OpenVINS sees anything:
- `cam_id` must be `< max_cameras`;
- width/height must equal the config `resolution` (×2 if `downsample_cameras`);
- pixel bytes must be exact, with no trailing bytes;
- every f64 must be finite.

## Responses
`0x81 ACK` (every request except `CAM`/`CAM_STEREO`): `u8 status`, then an optional UTF-8 message.

`0x83 STATE` (reply to `CAM`/`CAM_STEREO`):

| field | type |
|---|---|
| status | u8 |
| t (state time, IMU clock) | f64 (NaN if no state yet) |
| q_GtoI (JPL x,y,z,w) | 4 × f64 |
| p_IinG | 3 × f64 |
| v_IinG | 3 × f64 |
| cov | 36 × f64, row-major 6×6 marginal over **[p, θ]** |
| n_tracked | u32 (KLT tracks in cam 0's last frame) |
| n_msckf | u32 (MSCKF features used in the last update) |
| n_slam | u32 (SLAM features in the state) |
| n_list | u16 (≤ 200) |
| tracks | n_list × (`u32 id, f32 u, f32 v`), raw pixel coordinates |

When there is no state yet, the pose/cov fields are zeros and `t` is NaN.

## Status codes
| code | meaning |
|---|---|
| 0 `OK` | processed; the state is valid |
| 1 `NOT_INITIALIZED` | processed, but the estimator has no state yet (no `INIT_GT` and the dynamic init hasn't succeeded) |
| 2 `ERROR` | OpenVINS threw while processing |
| 3 `NEED_IMU` | frame not processed: IMU does not yet cover the camera time |
| 4 `BAD_REQUEST` | malformed or out-of-order request, or a bad config; nothing changed |

## Known limits (v1)
- OpenVINS still calls `std::exit` on some fatal config errors. The client must treat EOF as `VioFailure`.
- Reported tracks are cam 0 only.
- Not implemented yet (PLAN 1.1): clone-pair Δpose + joint covariance, and `fast_state_propagate` (PR #549).
