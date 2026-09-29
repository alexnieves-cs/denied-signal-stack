"""Codec for the ovserver lockstep protocol (cpp/ovserver/PROTOCOL.md, v1).

Frame: ``u32 payload_len | u8 type | payload`` little-endian.

Convention boundary (ADR-0002). OpenVINS reports ``q_GtoI`` as a JPL quaternion ``[x,y,z,w]``
with ``R_GtoI = R_jpl(q)``. Since ``R_jpl(q) = R_hamilton(q)^T`` for the same four numbers,
the Hamilton quaternion ``[w,x,y,z]`` of those numbers is ``q_ItoG`` (world <- IMU): a pure
reorder (:func:`dss.core.rotations.jpl_to_hamilton`).

Attitude error. OpenVINS's JPL update is ``q <- dq (x) q`` with ``dq ~ [dtheta/2, 1]``, i.e.
``R_GtoI_true = (I - [dtheta]x) R_GtoI_est``. Transposing:
``R_ItoG_true = R_ItoG_est (I + [dtheta]x)``, which is exactly our local (body-frame) Hamilton
error ``R_true = R_est exp([theta_local]x)``. So **theta_ours = +theta_JPL** to first order and
the covariance blocks need only a reorder, no sign flip or rotation (tested in
tests/vio/test_protocol.py by Monte Carlo).

Covariance orders:
- STATE reply 6x6: OpenVINS ``[p, theta]`` -> ours ``[p, theta]`` (identical).
- REANCHOR 15x15: ours ``[p, v, theta, bg, ba]`` -> OpenVINS ``[theta, p, v, bg, ba]``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from dss.core.rotations import hamilton_to_jpl, jpl_to_hamilton

INIT, IMU, CAM, INIT_GT, REANCHOR, SHUTDOWN, CAM_STEREO, PING = 1, 2, 3, 4, 5, 6, 7, 8
RSP_ACK, RSP_STATE = 0x81, 0x83
ST_OK, ST_NOT_INITIALIZED, ST_ERROR, ST_NEED_IMU, ST_BAD_REQUEST = 0, 1, 2, 3, 4
STATUS_NAMES = {0: "OK", 1: "NOT_INITIALIZED", 2: "ERROR", 3: "NEED_IMU", 4: "BAD_REQUEST"}
HDR = struct.Struct("<IB")
MAX_PAYLOAD = 64 << 20

# permutation: OpenVINS index for each of our 15 error-state indices
_OURS_TO_OV = np.r_[3:6, 6:9, 0:3, 9:12, 12:15]  # ours [p,v,th,bg,ba] -> ov position of each


class VioFailure(RuntimeError):
    """ovserver died, closed the pipe, or broke framing."""


@dataclass
class VioState:
    status: int
    t: float  # state time, IMU clock, seconds (nan if none)
    p: np.ndarray  # p_IinG (world)
    q: np.ndarray  # Hamilton [w,x,y,z], IMU -> world
    v: np.ndarray
    cov6: np.ndarray  # 6x6 over [p, theta_local]
    n_tracked: int = 0
    n_msckf: int = 0
    n_slam: int = 0
    tracks: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (id, u, v)

    @property
    def ok(self) -> bool:
        return self.status == ST_OK and np.isfinite(self.t)


def frame(typ: int, payload: bytes = b"") -> bytes:
    return HDR.pack(len(payload), typ) + payload


def enc_imu(t: float, w, a) -> bytes:
    return struct.pack("<7d", t, *w, *a)


def enc_cam(t: float, gray: np.ndarray, cam_id: int = 0) -> bytes:
    g = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = g.shape
    return struct.pack("<dBII", t, cam_id, w, h) + g.tobytes()


def enc_cam_stereo(t: float, g0: np.ndarray, g1: np.ndarray, ids=(0, 1)) -> bytes:
    out = [struct.pack("<d", t)]
    for cid, g in zip(ids, (g0, g1)):
        g = np.ascontiguousarray(g, dtype=np.uint8)
        h, w = g.shape
        out.append(struct.pack("<BII", cid, w, h) + g.tobytes())
    return b"".join(out)


def enc_state17(t: float, q_wxyz, p, v, bg, ba) -> bytes:
    qj = hamilton_to_jpl(np.asarray(q_wxyz, dtype=float))
    return struct.pack("<17d", t, *qj, *p, *v, *bg, *ba)


def cov15_ours_to_ov(P: np.ndarray) -> np.ndarray:
    """Reorder a 15x15 covariance from ours [p,v,th,bg,ba] to OpenVINS [th,p,v,bg,ba]."""
    perm = np.empty(15, dtype=int)  # perm[ov_index] = our_index
    perm[_OURS_TO_OV] = np.arange(15)
    return np.asarray(P)[np.ix_(perm, perm)]


def cov15_ov_to_ours(P: np.ndarray) -> np.ndarray:
    return np.asarray(P)[np.ix_(_OURS_TO_OV, _OURS_TO_OV)]


def enc_reanchor(t, q_wxyz, p, v, bg, ba, P15_ours: np.ndarray) -> bytes:
    P = cov15_ours_to_ov(0.5 * (P15_ours + P15_ours.T))
    return enc_state17(t, q_wxyz, p, v, bg, ba) + struct.pack("<225d", *P.ravel())


def dec_ack(payload: bytes) -> tuple[int, str]:
    if len(payload) < 1:
        raise VioFailure("empty ACK")
    return payload[0], payload[1:].decode(errors="replace")


def dec_state(b: bytes) -> VioState:
    if len(b) < 1 + 47 * 8 + 14:
        raise VioFailure(f"short STATE payload ({len(b)} bytes)")
    st = b[0]
    vals = np.frombuffer(b, dtype="<f8", count=47, offset=1)
    off = 1 + 47 * 8
    n_tracked, n_msckf, n_slam, n_list = struct.unpack_from("<IIIH", b, off)
    off += 14
    if len(b) != off + 12 * n_list:
        raise VioFailure("STATE track list length mismatch")
    tr = np.frombuffer(b, dtype=np.dtype([("id", "<u4"), ("u", "<f4"), ("v", "<f4")]), count=n_list, offset=off)
    tracks = np.stack([tr["id"].astype(float), tr["u"].astype(float), tr["v"].astype(float)], 1) if n_list else np.zeros((0, 3))
    t = float(vals[0])
    if np.isfinite(t):
        q = jpl_to_hamilton(vals[1:5])
    else:
        q = np.array([1.0, 0, 0, 0])
    return VioState(status=st, t=t, p=vals[5:8].copy(), q=q, v=vals[8:11].copy(), cov6=vals[11:47].reshape(6, 6).copy(),
                    n_tracked=n_tracked, n_msckf=n_msckf, n_slam=n_slam, tracks=tracks)
