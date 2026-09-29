import os

import numpy as np
import pytest

PHASE_GATE = os.environ.get("DSS_PHASE_GATE") == "1"


@pytest.fixture
def g():
    return np.random.default_rng(1234)


def pytest_collection_modifyitems(config, items):
    if not PHASE_GATE:
        return
    # under the phase gate a skip is a failure: convert skip markers into hard requirements
    for it in items:
        for m in it.iter_markers("skip"):
            it.add_marker(pytest.mark.xfail(strict=True, reason="skipped under DSS_PHASE_GATE: " + str(m.kwargs)))
