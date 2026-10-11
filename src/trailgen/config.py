"""Pipeline parameters, loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.toml"


@dataclass
class GridCfg:
    dem_resolution: float = 1.0
    chm_resolution: float = 0.5
    cost_resolution: float = 0.25


@dataclass
class GroundCfg:
    classification: int = 2
    min_ground_fraction: float = 0.01
    max_ground_residual: float = 0.5


@dataclass
class TreesCfg:
    min_tree_height: float = 3.0
    min_crown_height: float = 2.0
    window_base: float = 0.6
    window_slope: float = 0.08
    max_crown_radius: float = 10.0
    min_saddle_drop: float = 1.5
    min_saddle_fraction: float = 0.15
    dbh_a: float = 1.5
    dbh_b: float = 1.05


@dataclass
class VegetationCfg:
    low_min: float = 0.3
    low_max: float = 2.0
    smoothing: float = 1.0


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
class Config:
    grid: GridCfg = field(default_factory=GridCfg)
    ground: GroundCfg = field(default_factory=GroundCfg)
    trees: TreesCfg = field(default_factory=TreesCfg)
    vegetation: VegetationCfg = field(default_factory=VegetationCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    cost_weights: CostWeightsCfg = field(default_factory=CostWeightsCfg)

    def to_dict(self) -> dict:
        return asdict(self)


def _apply(section, values: dict, name: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key [{name}] {key}")
        setattr(section, key, type(getattr(section, key))(value))


def load_config(path: str | Path | None = None) -> Config:
    """Defaults from configs/default.toml (if present), then overrides from `path`."""
    cfg = Config()
    for p in (DEFAULT_CONFIG, path):
        if p is None or not Path(p).exists():
            continue
        with open(p, "rb") as fh:
            data = tomllib.load(fh)
        for name, values in data.items():
            if not hasattr(cfg, name):
                raise KeyError(f"Unknown config section [{name}]")
            _apply(getattr(cfg, name), values, name)
    return cfg
