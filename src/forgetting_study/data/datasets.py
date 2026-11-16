import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from forgetting_study.config import TaskConfig
from forgetting_study.data import instances

IGNORE_INDEX = -100

# Baseline settings of the MAD benchmark (configs/tasks/*.yml in MAD-lab).
TASKS = {
    "in-context-recall": dict(
        fn=instances.generate_in_context_recall_instance,
        recall=True,
        num_train_examples=12_800,
        defaults=dict(vocab_size=16, seq_len=128, multi_query=True, noise_vocab_size=0, frac_noise=0.0),
    ),
    "noisy-in-context-recall": dict(
        fn=instances.generate_noisy_in_context_recall_instance,
        recall=True,
        num_train_examples=12_800,
        defaults=dict(vocab_size=32, seq_len=128, multi_query=True, noise_vocab_size=16, frac_noise=0.2),
    ),
    "fuzzy-in-context-recall": dict(
        fn=instances.generate_fuzzy_in_context_recall_instance,
        recall=True,
        num_train_examples=12_800,
        defaults=dict(
            vocab_size=16, seq_len=128, multi_query=True, k_motif_size=3, v_motif_size=3,
            noise_vocab_size=0, frac_noise=0.0,
        ),
    ),
    "selective-copying": dict(
        fn=instances.generate_selective_copying_instance,
        recall=False,
        num_train_examples=12_800,
        defaults=dict(vocab_size=16, seq_len=256, num_tokens_to_copy=16),
    ),
    "memorization": dict(
        fn=instances.generate_memorization_instance,
        recall=False,
        num_train_examples=256,
        defaults=dict(vocab_size=256, seq_len=32, noise_vocab_size=0, frac_noise=0.0),
    ),
}


@dataclass
class TaskData:
    name: str
    vocab_size: int
    seq_len: int
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    # Targets keyed by loss kind: "masked" always, "all" for recall tasks.
    test_y: dict[str, torch.Tensor]


def generator_kwargs(cfg: TaskConfig) -> dict:
    if cfg.name not in TASKS:
        raise ValueError(f"unknown task {cfg.name!r}; choose from {sorted(TASKS)}")
    return {**TASKS[cfg.name]["defaults"], **cfg.overrides}


def all_token_targets(x: np.ndarray, y_masked: np.ndarray) -> np.ndarray:
    """Next-token targets for every position of a recall sequence.

    Recall inputs are seq[:-1] and the final masked target is always seq[-1],
    so the shifted full sequence can be rebuilt from (inputs, masked targets).
    """
    assert (y_masked[:, -1] != IGNORE_INDEX).all()
    return np.concatenate([x[:, 1:], y_masked[:, -1:]], axis=1)


def _generate_split(cfg: TaskConfig, kwargs: dict, num_examples: int, is_training: bool, seed: int):
    task = TASKS[cfg.name]
    rng = np.random.default_rng(seed)
    call_kwargs = dict(kwargs, rng=rng, is_training=is_training, target_ignore_idx=IGNORE_INDEX)
    if task["recall"]:
        call_kwargs["masked_targets"] = True
    if cfg.name == "memorization":
        call_kwargs["kv_map"] = instances.generate_kv_map(
            vocab_size=kwargs["vocab_size"] - 1 - kwargs.get("noise_vocab_size", 0), seed=cfg.seed
        )
    split = "train" if is_training else "test"
    xs, ys = [], []
    for _ in tqdm(range(num_examples), desc=f"generating {cfg.name} {split}", leave=False):
        x, y = task["fn"](**call_kwargs)
        xs.append(x)
        ys.append(y)
    return np.stack(xs).astype(np.int64), np.stack(ys).astype(np.int64)


def _warn_on_leakage(train_x: np.ndarray, test_x: np.ndarray) -> None:
    train_set = {row.tobytes() for row in train_x}
    frac = np.mean([row.tobytes() in train_set for row in test_x])
    if frac > 0.001:
        print(f"WARNING: {frac:.1%} of test inputs also appear in the training set")


def load_task_data(cfg: TaskConfig, device: str | torch.device = "cpu") -> TaskData:
    """Generate (or load cached) train/test data for a MAD task."""
    task = TASKS.get(cfg.name)
    kwargs = generator_kwargs(cfg)
    num_train = cfg.num_train_examples or task["num_train_examples"]
    if cfg.train_targets not in ("masked", "all"):
        raise ValueError(f"train_targets must be 'masked' or 'all', got {cfg.train_targets!r}")
    if cfg.train_targets == "all" and not task["recall"]:
        raise ValueError(f"train_targets='all' is only defined for recall tasks, not {cfg.name!r}")

    key = json.dumps(
        dict(task=cfg.name, kwargs=kwargs, num_train=num_train, num_test=cfg.num_test_examples, seed=cfg.seed),
        sort_keys=True,
    )
    cache = Path(cfg.data_dir) / f"{cfg.name}-{hashlib.sha1(key.encode()).hexdigest()[:12]}.npz"
    if cache.exists():
        arrays = dict(np.load(cache))
    else:
        train_x, train_y = _generate_split(cfg, kwargs, num_train, is_training=True, seed=cfg.seed)
        test_x, test_y = _generate_split(cfg, kwargs, cfg.num_test_examples, is_training=False, seed=cfg.seed + 1)
        _warn_on_leakage(train_x, test_x)
        arrays = dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **arrays)
        cache.with_suffix(".json").write_text(key)

    train_y = arrays["train_y"]
    test_y = {"masked": arrays["test_y"]}
    if task["recall"]:
        test_y["all"] = all_token_targets(arrays["test_x"], arrays["test_y"])
        if cfg.train_targets == "all":
            train_y = all_token_targets(arrays["train_x"], train_y)

    to_t = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
    return TaskData(
        name=cfg.name,
        vocab_size=kwargs["vocab_size"],
        seq_len=arrays["train_x"].shape[1],
        train_x=to_t(arrays["train_x"]),
        train_y=to_t(train_y),
        test_x=to_t(arrays["test_x"]),
        test_y={k: to_t(v) for k, v in test_y.items()},
    )
