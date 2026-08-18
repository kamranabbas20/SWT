"""SWT -- a smart seismic-to-well tie.

A deterministic geophysics core with an AI layer on top.  The core is exact and
testable; the AI layer makes it fast and honest, and is never allowed to make it
"correct" by optimising a correlation coefficient.

The usual sequence::

    from swt.io.las import load_las
    from swt.io.segy import survey_info, extract_trace
    from swt.timedepth.integrate import integrate_sonic, ShallowModel
    from swt.timedepth.calibrate import calibrate_to_checkshots
    from swt.tie.iterate import tie

    logs = load_las("well.las")
    seismic = extract_trace("volume.sgy", x=..., y=...)

    raw = integrate_sonic(depth_tvdss, logs.sonic_us_per_m,
                          ShallowModel(replacement_velocity=1900.0))
    calibrated = calibrate_to_checkshots(raw, cs_depth, cs_twt).time_depth

    result = tie(seismic, calibrated, logs.sonic_us_per_m, logs.density_g_cm3)
    print(result.summary())

With no data to hand, ``swt.forward.make_case`` builds a complete tie problem
that knows its own answer.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .trace import Trace
from .timedepth.model import TimeDepth
from .wavelet.core import Wavelet

__all__ = ["Trace", "TimeDepth", "Wavelet", "__version__"]
