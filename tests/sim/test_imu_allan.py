import numpy as np
import pytest

from dss.core import config
from dss.sim import imu, trajectory


def allan_var(x, dt, taus):
    out = []
    c = np.concatenate([[0], np.cumsum(x) * dt])
    for tau in taus:
        m = int(round(tau / dt))
        n = len(x)
        # overlapping Allan variance from the integrated signal
        th = c
        d = th[2 * m:] - 2 * th[m:-m] + th[: -2 * m]
        out.append(np.sum(d**2) / (2 * (m * dt) ** 2 * (n - 2 * m + 1)))
    return np.array(out)


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["sensors/imu_bmi160.yaml", "sensors/imu_adis16448.yaml"])
def test_allan_fit(preset):
    c = config.load(preset)
    dt = 1.0 / c["rate_hz"]
    n = int(3 * 3600 / dt)
    taus = np.logspace(np.log10(0.05), np.log10(1080), 30)
    d = imu.simulate_imu(trajectory.static(3 * 3600 + 1.0), imu.ImuParams.from_config(c), 7, t0=0.0, t1=n * dt)
    for kind, N, K in [("gyro", c["gyroscope_noise_density"], c["gyroscope_random_walk"]),
                       ("acc", c["accelerometer_noise_density"], c["accelerometer_random_walk"])]:
        for ax in range(3):
            x = d[kind][:, ax]
            av = allan_var(x, dt, taus)
            A = np.stack([1 / taus, taus / 3], 1)
            w = 1 / av  # WLS in relative terms
            coef, *_ = np.linalg.lstsq(A * w[:, None], av * w, rcond=None)
            Nh, Kh = np.sqrt(coef[0]), np.sqrt(coef[1])
            assert abs(Nh / N - 1) < 0.05, (kind, ax, Nh, N)
            assert abs(Kh / K - 1) < 0.20, (kind, ax, Kh, K)
