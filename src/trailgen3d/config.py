"""TrailGen3D parameters: the [trailgen3d.*] tables of a TOML config file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from sky2trail.config import DEFAULT_CONFIG, apply_table, read_toml


@dataclass
class GridCfg:
    cost_resolution: float = 0.25


@dataclass
class RobotCfg:
    half_width: float = 0.16
    safety_margin: float = 0.10
    max_slope: float = 30.0
    max_roughness: float = 0.15


@dataclass
class CostWeightsCfg:
    slope: float = 0.5
    roughness: float = 0.2
    shrub: float = 0.6


@dataclass
class WorldConfig:
    grid: GridCfg = field(default_factory=GridCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    cost_weights: CostWeightsCfg = field(default_factory=CostWeightsCfg)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> WorldConfig:
    """Defaults, then configs/default.toml, then `path` (each only its [trailgen3d] tables)."""
    cfg = WorldConfig()
    for p in (DEFAULT_CONFIG, path):
        apply_table(cfg, read_toml(p).get("trailgen3d", {}), "trailgen3d")
    return cfg
