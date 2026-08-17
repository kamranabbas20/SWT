"""SEG-Y reading and trace extraction at a well.

The recurring problem with SEG-Y is not the format, it is that **trace headers
lie**.  Coordinate scalars are wrong or missing, X and Y are swapped, coordinates
are in arc-seconds when the header says metres, or the survey geometry simply
was not written.  A reader that trusts the headers will place the well hundreds
of metres from where it is and extract a trace from the wrong place -- and the
tie that follows will look merely mediocre rather than obviously broken, which
is the worst possible failure mode.

So the geometry here is always *reported* before it is used, and always
*overridable*.  :func:`survey_info` exists to be looked at by a human before any
extraction happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..trace import Trace


@dataclass(frozen=True)
class SurveyInfo:
    """What a SEG-Y file claims about itself."""

    path: str
    n_traces: int
    n_samples: int
    dt_s: float
    t0_s: float
    coordinate_scalar: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    inline_range: tuple[int, int] | None
    xline_range: tuple[int, int] | None
    format_code: int

    def summary(self) -> dict:
        return {
            "path": self.path,
            "n_traces": self.n_traces,
            "n_samples": self.n_samples,
            "dt_ms": round(self.dt_s * 1e3, 3),
            "time_range_ms": [
                round(self.t0_s * 1e3, 1),
                round((self.t0_s + (self.n_samples - 1) * self.dt_s) * 1e3, 1),
            ],
            "coordinate_scalar": self.coordinate_scalar,
            "x_range": [round(v, 2) for v in self.x_range],
            "y_range": [round(v, 2) for v in self.y_range],
            "inline_range": list(self.inline_range) if self.inline_range else None,
            "xline_range": list(self.xline_range) if self.xline_range else None,
            "format_code": self.format_code,
            "warnings": self.warnings(),
        }

    def warnings(self) -> list[str]:
        """Geometry problems worth a human's attention before extracting."""
        issues = []
        if self.coordinate_scalar == 0:
            issues.append(
                "Coordinate scalar is 0, which is not a legal SEG-Y value; SWT is "
                "treating it as 1. Verify the coordinates against a known well location."
            )
        span_x = self.x_range[1] - self.x_range[0]
        span_y = self.y_range[1] - self.y_range[0]
        if span_x == 0 and span_y == 0:
            issues.append(
                "All traces report the same coordinate: the geometry is missing from "
                "the headers. Extraction by position is impossible -- extract by "
                "inline/crossline, or supply coordinates explicitly."
            )
        if max(abs(v) for v in self.x_range + self.y_range) < 1000:
            issues.append(
                "Coordinates are small numbers; they may be in kilometres, arc-seconds "
                "or a local grid rather than metres. Check before trusting distances."
            )
        return issues


def survey_info(path: str) -> SurveyInfo:
    """Inspect a SEG-Y file without loading the data.

    Call this first, every time, on any file that has not been checked before.
    """
    import segyio

    with segyio.open(path, "r", ignore_geometry=True) as handle:
        n_traces = handle.tracecount
        samples = np.asarray(handle.samples, dtype=float)
        dt_s = float(samples[1] - samples[0]) / 1000.0 if samples.size > 1 else 0.0
        t0_s = float(samples[0]) / 1000.0

        header = handle.header[0]
        scalar = int(header[segyio.TraceField.SourceGroupScalar])

        x, y = _coordinates(handle, scalar)
        inlines = _header_range(handle, segyio.TraceField.INLINE_3D)
        xlines = _header_range(handle, segyio.TraceField.CROSSLINE_3D)
        format_code = int(handle.format)

    return SurveyInfo(
        path=path,
        n_traces=n_traces,
        n_samples=int(samples.size),
        dt_s=dt_s,
        t0_s=t0_s,
        coordinate_scalar=scalar,
        x_range=(float(np.min(x)), float(np.max(x))),
        y_range=(float(np.min(y)), float(np.max(y))),
        inline_range=inlines,
        xline_range=xlines,
        format_code=format_code,
    )


def apply_coordinate_scalar(values: np.ndarray, scalar: int) -> np.ndarray:
    """Apply the SEG-Y coordinate scalar convention.

    Positive multiplies, negative divides, and zero is illegal but common --
    treated as 1, with :meth:`SurveyInfo.warnings` saying so out loud.
    """
    if scalar > 0:
        return values * float(scalar)
    if scalar < 0:
        return values / float(-scalar)
    return values.astype(float)


def extract_trace(
    path: str,
    x: float | None = None,
    y: float | None = None,
    inline: int | None = None,
    xline: int | None = None,
    aperture: int = 1,
    label: str = "seismic",
) -> Trace:
    """Extract a trace at a well location, by coordinate or by inline/crossline.

    Parameters
    ----------
    x, y:
        Well surface coordinates in the survey's own coordinate system.  The
        nearest trace is used.
    inline, xline:
        Alternative addressing, and the reliable one when the coordinate headers
        are not trustworthy.
    aperture:
        Number of traces per side to average, e.g. ``3`` averages a 3x3 patch
        centred on the well.  Averaging lifts signal-to-noise and also smooths
        genuine detail, so it defaults to 1 (no averaging) and should be a
        deliberate choice.

    Notes
    -----
    Averaging here is by nearest neighbours in trace *index* order, which is a
    true spatial neighbourhood only when the file is sorted into a regular
    geometry.  For an unsorted or 2D file, keep ``aperture=1``.
    """
    import segyio

    if (x is None) != (y is None):
        raise ValueError("give both x and y, or neither")
    if (inline is None) != (xline is None):
        raise ValueError("give both inline and xline, or neither")
    if (x is None) == (inline is None):
        raise ValueError("address the trace either by x/y or by inline/xline, not both or neither")
    if aperture < 1:
        raise ValueError("aperture must be at least 1")

    with segyio.open(path, "r", ignore_geometry=True) as handle:
        samples = np.asarray(handle.samples, dtype=float)
        if samples.size < 2:
            raise ValueError(f"{path} has fewer than two time samples")
        twt = samples / 1000.0

        if x is not None:
            scalar = int(handle.header[0][segyio.TraceField.SourceGroupScalar])
            trace_x, trace_y = _coordinates(handle, scalar)
            distance = np.hypot(trace_x - float(x), trace_y - float(y))
            index = int(np.argmin(distance))
            chosen_distance = float(distance[index])
            addressing = {"nearest_trace_distance": round(chosen_distance, 2)}
        else:
            index = _find_by_line(handle, int(inline), int(xline))
            chosen_distance = 0.0
            addressing = {"inline": int(inline), "xline": int(xline)}

        indices = _aperture_indices(index, aperture, handle.tracecount)
        stack = np.mean([np.asarray(handle.trace[i], dtype=float) for i in indices], axis=0)

    return Trace(
        twt=twt,
        amplitude=stack,
        label=label,
        meta={
            "path": path,
            "trace_index": index,
            "n_traces_stacked": len(indices),
            "aperture": aperture,
            **addressing,
        },
    )


def _coordinates(handle, scalar: int) -> tuple[np.ndarray, np.ndarray]:
    import segyio

    x = np.asarray(handle.attributes(segyio.TraceField.CDP_X)[:], dtype=float)
    y = np.asarray(handle.attributes(segyio.TraceField.CDP_Y)[:], dtype=float)
    if not np.any(x) and not np.any(y):
        # Fall back to source coordinates, which some vintages populate instead.
        x = np.asarray(handle.attributes(segyio.TraceField.SourceX)[:], dtype=float)
        y = np.asarray(handle.attributes(segyio.TraceField.SourceY)[:], dtype=float)
    return apply_coordinate_scalar(x, scalar), apply_coordinate_scalar(y, scalar)


def _header_range(handle, field) -> tuple[int, int] | None:
    values = np.asarray(handle.attributes(field)[:], dtype=int)
    if not values.size or (values.min() == 0 and values.max() == 0):
        return None
    return int(values.min()), int(values.max())


def _find_by_line(handle, inline: int, xline: int) -> int:
    import segyio

    inlines = np.asarray(handle.attributes(segyio.TraceField.INLINE_3D)[:], dtype=int)
    xlines = np.asarray(handle.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=int)
    match = np.flatnonzero((inlines == inline) & (xlines == xline))
    if not match.size:
        raise KeyError(
            f"no trace at inline {inline}, crossline {xline}; "
            f"file covers inlines {inlines.min()}-{inlines.max()}, "
            f"crosslines {xlines.min()}-{xlines.max()}"
        )
    return int(match[0])


def _aperture_indices(index: int, aperture: int, n_traces: int) -> list[int]:
    if aperture <= 1:
        return [index]
    half = aperture // 2
    lo = max(0, index - half)
    hi = min(n_traces, index + half + 1)
    return list(range(lo, hi))
