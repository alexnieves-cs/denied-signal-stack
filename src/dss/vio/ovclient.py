"""Lockstep client for ovserver (GPL-3.0 OpenVINS process). EOF or a broken frame -> VioFailure."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from dss.vio import protocol as pr
from dss.vio.protocol import VioFailure, VioState

BIN_DIR = Path.home() / ".cache/dss/cpp/bin"
DEFAULT_BINARY = BIN_DIR / "ovserver-release"


class OvClient:
    def __init__(self, config_path: str | Path, binary: str | Path | None = None, stderr_path: str | Path | None = None,
                 verbosity: str | None = None):
        self.config_path = str(Path(config_path).resolve())
        self.binary = str(binary or os.environ.get("DSS_OVSERVER", DEFAULT_BINARY))
        self._err = open(stderr_path or os.devnull, "wb")
        self.proc = subprocess.Popen([self.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, bufsize=0)
        self.last_imu_t = -np.inf
        self.framing_errors = 0

    # --- transport -------------------------------------------------------------------
    def _read_exact(self, n: int) -> bytes:
        assert self.proc.stdout is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise VioFailure(f"ovserver closed the pipe (exit={self.proc.poll()})")
            buf += chunk
        return bytes(buf)

    def request(self, typ: int, payload: bytes = b"") -> tuple[int, bytes]:
        if self.proc.poll() is not None:
            raise VioFailure(f"ovserver exited with {self.proc.returncode}")
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(pr.frame(typ, payload))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise VioFailure(f"ovserver pipe broken: {e}") from e
        n, rtyp = pr.HDR.unpack(self._read_exact(pr.HDR.size))
        if n > pr.MAX_PAYLOAD or rtyp not in (pr.RSP_ACK, pr.RSP_STATE):
            self.framing_errors += 1
            raise VioFailure(f"framing error: type={rtyp:#x} len={n}")
        return rtyp, self._read_exact(n)

    def _ack(self, typ: int, payload: bytes = b"") -> tuple[int, str]:
        rtyp, b = self.request(typ, payload)
        if rtyp != pr.RSP_ACK:
            self.framing_errors += 1
            raise VioFailure(f"expected ACK, got {rtyp:#x}")
        return pr.dec_ack(b)

    def _check(self, what: str, st: int, msg: str) -> None:
        if st != pr.ST_OK:
            raise ValueError(f"ovserver rejected {what}: {pr.STATUS_NAMES.get(st, st)} {msg}")

    # --- API ---------------------------------------------------------------------------
    def ping(self) -> str:
        st, msg = self._ack(pr.PING)
        self._check("PING", st, msg)
        return msg

    def init(self) -> None:
        self._check("INIT", *self._ack(pr.INIT, self.config_path.encode()))
        self.last_imu_t = -np.inf

    def imu(self, t_s: float, w, a) -> None:
        st, msg = self._ack(pr.IMU, pr.enc_imu(t_s, w, a))
        self._check("IMU", st, msg)
        self.last_imu_t = t_s

    def cam(self, t_s: float, gray: np.ndarray, cam_id: int = 0) -> VioState:
        rtyp, b = self.request(pr.CAM, pr.enc_cam(t_s, gray, cam_id))
        if rtyp != pr.RSP_STATE:
            self.framing_errors += 1
            raise VioFailure("expected STATE")
        return pr.dec_state(b)

    def cam_stereo(self, t_s: float, g0: np.ndarray, g1: np.ndarray) -> VioState:
        rtyp, b = self.request(pr.CAM_STEREO, pr.enc_cam_stereo(t_s, g0, g1))
        if rtyp != pr.RSP_STATE:
            self.framing_errors += 1
            raise VioFailure("expected STATE")
        return pr.dec_state(b)

    def init_gt(self, t_s, q_wxyz, p, v, bg=(0, 0, 0), ba=(0, 0, 0)) -> None:
        self._check("INIT_GT", *self._ack(pr.INIT_GT, pr.enc_state17(t_s, q_wxyz, p, v, bg, ba)))

    def reanchor(self, t_s, q_wxyz, p, v, bg, ba, P15: np.ndarray) -> None:
        """P15 in OUR order [p, v, theta_local, bg, ba]."""
        self._check("REANCHOR", *self._ack(pr.REANCHOR, pr.enc_reanchor(t_s, q_wxyz, p, v, bg, ba, np.asarray(P15))))

    def rss_mb(self) -> float:
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.proc.pid)], capture_output=True, text=True).stdout
            return int(out.strip()) / 1024.0
        except Exception:
            return float("nan")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._ack(pr.SHUTDOWN)
            except (VioFailure, ValueError):
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._err.close()

    def __enter__(self) -> OvClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
