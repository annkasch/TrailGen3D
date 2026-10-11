"""What a tile can resolve: point density, returns per pulse, ground share."""

from __future__ import annotations

import numpy as np

from .io import PointCloud


def density_report(pc: PointCloud) -> dict:
    xmin, ymin, xmax, ymax = pc.bounds
    area = max((xmax - xmin) * (ymax - ymin), 1e-9)
    first = pc.return_number == 1
    pulses = int(first.sum())
    pulse_density = pulses / area
    ground = pc.classification == 2
    classes, counts = np.unique(pc.classification, return_counts=True)
    spacing = 1.0 / np.sqrt(pulse_density) if pulse_density > 0 else float("nan")
    # First returns approximate pulses; allow ~10 % for pulses that returned nothing.
    if pulse_density >= 7.2:
        level = "about QL1 or better (nominal >= 8 pulses/m2)"
    elif pulse_density >= 1.8:
        level = "about QL2 (nominal >= 2 pulses/m2)"
    else:
        level = "below QL2 (nominal < 2 pulses/m2)"
    return {
        "points": int(len(pc)),
        "area_m2": round(area, 1),
        "points_per_m2": round(len(pc) / area, 2),
        "first_returns_per_m2": round(pulse_density, 2),
        "mean_pulse_spacing_m": round(spacing, 2),
        "returns_per_pulse": round(len(pc) / max(pulses, 1), 2),
        "ground_fraction": round(float(ground.mean()), 3),
        "classes": {int(c): int(n) for c, n in zip(classes, counts)},
        "estimated_quality_level": level,
        "smallest_reliable_feature_m": round(2 * spacing, 2),
    }
