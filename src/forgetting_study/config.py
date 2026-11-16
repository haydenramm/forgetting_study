from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class TaskConfig:
    name: str = "in-context-recall"
    # Generator kwargs that override the MAD benchmark defaults for the task
    # (e.g. vocab_size, seq_len, frac_noise); see data/datasets.py.
    overrides: dict = field(default_factory=dict)
    num_train_examples: int | None = None
    num_test_examples: int = 1_280
    # "masked": loss only on predictable tokens. "all": next-token loss on every
    # token (only defined for the recall tasks).
    train_targets: str = "masked"
    seed: int = 0
    data_dir: str = "data"


@dataclass
class ModelConfig:
    n_blocks: int = 1
    dim: int = 128
    mlp_ratio: int = 4
    residual: bool = False
    norm: bool = False
    bias: bool = False


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 5e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: float | None = None
    eval_every_epochs: int = 1
    device: str = "cuda"
    seed: int = 0


@dataclass
class RecordConfig:
    enabled: bool = True
    # Record every `early_every_steps` until `early_until_step`, then every `every_steps`.
    early_every_steps: int = 10
    early_until_step: int = 500
    every_steps: int = 500
    probe_size: int = 4
    dtype: str = "bfloat16"


@dataclass
class RunConfig:
    name: str | None = None
    out_dir: str = "runs"
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    record: RecordConfig = field(default_factory=RecordConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, values: dict):
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for key, value in values.items():
        default = known[key].default_factory() if callable(known[key].default_factory) else None
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _set_dotted(d: dict, dotted_key: str, value) -> None:
    *parents, leaf = dotted_key.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


def load_config(path: str | Path | None = None, overrides: list[str] = ()) -> RunConfig:
    """Load a RunConfig from YAML, applying `section.key=value` overrides."""
    values = {}
    if path is not None:
        with open(path) as f:
            values = yaml.safe_load(f) or {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        _set_dotted(values, key, yaml.safe_load(raw))
    return _build(RunConfig, values)
