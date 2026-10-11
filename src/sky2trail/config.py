"""Sky2Trail parameters: the [sky2trail.*] tables of a TOML config file."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

# Repository default; absent in a non-editable install, where dataclass defaults apply.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.toml"


@dataclass
class GridCfg:
    dem_resolution: float = 1.0
    chm_resolution: float = 0.5


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


@dataclass
class StemsV0Cfg:
    """Rules the v0 baseline uses to guess stems. PLACEHOLDERS, to be fitted to field data."""

    dbh_a: float = 1.5               # dbh_cm = dbh_a * height_m ** dbh_b
    dbh_b: float = 1.05
    crown_base_fraction: float = 0.4  # crown base at this share of tree height
    point_spacing: float = 0.1        # spacing of generated stem points, meters


@dataclass
class VegetationCfg:
    low_min: float = 0.3
    low_max: float = 2.0
    smoothing: float = 1.0


@dataclass
class PatchesCfg:
    """Unit of work for trained models: predicted core plus a context margin."""

    core_size: float = 10.0  # side of the square the model predicts, meters
    margin: float = 6.0      # extra aerial context on every side (about one crown radius)


@dataclass
class Stage1Cfg:
    """Stage 1 (learned stem map). Stored in each checkpoint so prediction matches training."""

    resolution: float = 0.25          # input/output pixel size, meters
    # Height bands (m above ground) for the vertical profile of non-ground returns;
    # the last band is open-ended.
    band_edges: list = field(default_factory=lambda: [0.0, 0.3, 2.0, 5.0, 10.0, 20.0])
    height_scale: float = 30.0        # heights are divided by this before the network
    density_radius: float = 2.0       # neighbourhood for the local pulse-density channel
    small_dbh: float = 0.10           # stems below this go to the 'small stem' heatmap
    heatmap_sigma: float = 0.5        # width of the target blob per stem, meters
    peak_threshold: float = 0.3       # minimum heatmap value of a predicted stem
    peak_radius: float = 1.0          # two predicted stems are at least this far apart


@dataclass
class Sky2TrailConfig:
    grid: GridCfg = field(default_factory=GridCfg)
    ground: GroundCfg = field(default_factory=GroundCfg)
    trees: TreesCfg = field(default_factory=TreesCfg)
    stems_v0: StemsV0Cfg = field(default_factory=StemsV0Cfg)
    vegetation: VegetationCfg = field(default_factory=VegetationCfg)
    patches: PatchesCfg = field(default_factory=PatchesCfg)
    stage1: Stage1Cfg = field(default_factory=Stage1Cfg)

    def to_dict(self) -> dict:
        return asdict(self)


def read_toml(path: str | Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def apply_table(cfg, table: dict, where: str) -> None:
    """Copy values from a TOML table onto a (nested) dataclass, rejecting unknown keys."""
    known = {f.name for f in fields(cfg)}
    for key, value in table.items():
        if key not in known:
            raise KeyError(f"Unknown config key [{where}] {key}")
        current = getattr(cfg, key)
        if is_dataclass(current):
            apply_table(current, value, f"{where}.{key}")
        else:
            setattr(cfg, key, type(current)(value))


def load_config(path: str | Path | None = None) -> Sky2TrailConfig:
    """Defaults, then configs/default.toml, then `path` (each only its [sky2trail] tables)."""
    cfg = Sky2TrailConfig()
    for p in (DEFAULT_CONFIG, path):
        apply_table(cfg, read_toml(p).get("sky2trail", {}), "sky2trail")
    return cfg
