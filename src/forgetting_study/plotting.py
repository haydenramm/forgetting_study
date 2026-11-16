"""Heatmaps of recorded parameter gradients, activations and activation gradients.

A "field" selects one group of recorded tensors from a snapshot:
  param_grads/train_batch, param_grads/probe_masked, param_grads/probe_all,
  activations, activation_grads/masked, activation_grads/all
Activation fields have a leading probe-example dimension, selected with `example`.
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm


class Run:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.snapshot_paths = sorted((self.dir / "snapshots").glob("step_*.pt"))
        self.steps = [int(p.stem.split("_")[1]) for p in self.snapshot_paths]
        metrics_path = self.dir / "metrics.jsonl"
        self.metrics = [json.loads(l) for l in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
        self._cache: dict[int, dict] = {}

    def snapshot(self, step: int) -> dict:
        """Snapshot at the recorded step closest to `step` (-1 for the last)."""
        i = len(self.steps) - 1 if step < 0 else int(np.argmin([abs(s - step) for s in self.steps]))
        if i not in self._cache:
            self._cache = {i: torch.load(self.snapshot_paths[i], weights_only=True)}
        return self._cache[i]

    def iter_snapshots(self):
        for path in self.snapshot_paths:
            yield torch.load(path, weights_only=True)

    def fields(self) -> list[str]:
        snap = self.snapshot(0)
        return ([f"param_grads/{k}" for k in snap["param_grads"]] + ["activations"]
                + [f"activation_grads/{k}" for k in snap["activation_grads"]])


def get_field(snap: dict, field: str, example: int = 0) -> dict[str, np.ndarray]:
    group, _, sub = field.partition("/")
    tensors = snap[group][sub] if sub else snap[group]
    out = {}
    for name, t in tensors.items():
        t = t.float()
        if group != "param_grads":
            t = t[example]
        out[name] = t.numpy()
    return out


def _as_2d(name: str, a: np.ndarray, mask_future: bool = True) -> np.ndarray:
    if a.ndim == 1:
        return a[None, :]
    if mask_future and ".attn." in name and a.shape[0] == a.shape[1]:
        a = a.copy()
        a[np.triu_indices(a.shape[0], 1)] = np.nan
    return a


def _symmetric_limit(a: np.ndarray, q: float = 0.995) -> float:
    finite = np.abs(a[np.isfinite(a)])
    lim = float(np.quantile(finite, q)) if finite.size else 0.0
    return lim if lim > 0 else 1.0


def heatmap(ax, name: str, a: np.ndarray, title: str | None = None, vlim: float | None = None):
    a = _as_2d(name, a)
    if name.endswith("attn.probs") and not np.any(a[np.isfinite(a)] < 0):
        im = ax.imshow(a, cmap="viridis", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    else:
        vlim = vlim or _symmetric_limit(a)
        im = ax.imshow(a, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto", interpolation="nearest")
    ax.set_title(title or f"{name}  {tuple(a.shape)}", fontsize=8)
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    return im


def _grid(n: int, ncols: int = 4, size: float = 3.6):
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * 0.85 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def plot_snapshot(run: Run, field: str, step: int, example: int = 0) -> plt.Figure:
    """One heatmap per tensor in `field` at a single recorded step."""
    snap = run.snapshot(step)
    tensors = get_field(snap, field, example)
    fig, axes = _grid(len(tensors))
    for ax, (name, a) in zip(axes, tensors.items()):
        heatmap(ax, name, a)
    suffix = "" if field.startswith("param_grads") else f", probe example {example}"
    fig.suptitle(f"{run.dir.name}: {field} at step {snap['step']} (epoch {snap['epoch']}){suffix}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_metrics(run: Run) -> plt.Figure:
    epochs = [m["epoch"] for m in run.metrics]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.5))
    for key in [k for k in run.metrics[0] if "loss" in k]:
        ax_loss.plot(epochs, [m[key] for m in run.metrics], label=key)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend(fontsize=8)
    ax_acc.plot(epochs, [m["test/acc"] for m in run.metrics])
    ax_acc.set_ylim(0, 1)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_title("test accuracy (predictable tokens)", fontsize=9)
    fig.suptitle(run.dir.name, fontsize=10)
    fig.tight_layout()
    return fig


def plot_norms(run: Run, field: str, example: int | None = None) -> plt.Figure:
    """log10 Frobenius norm of every tensor in `field` (rows) over recorded steps (columns).

    For activation fields the norm is over all probe examples unless `example` is given.
    """
    names, columns = None, []
    for snap in run.iter_snapshots():
        group, _, sub = field.partition("/")
        tensors = snap[group][sub] if sub else snap[group]
        names = names or list(tensors)
        col = []
        for n in names:
            t = tensors[n].float()
            if example is not None and group != "param_grads":
                t = t[example]
            col.append(t.norm().item())
        columns.append(col)
    norms = np.log10(np.maximum(np.array(columns).T, 1e-30))
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(names) + 1.5))
    im = ax.imshow(norms, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(len(names)), names, fontsize=7)
    xt = np.linspace(0, len(run.steps) - 1, min(10, len(run.steps))).astype(int)
    ax.set_xticks(xt, [run.steps[i] for i in xt], fontsize=7)
    ax.set_xlabel("step (one column per snapshot)")
    plt.colorbar(im, ax=ax, label="log10 norm")
    ax.set_title(f"{run.dir.name}: {field} norms", fontsize=10)
    fig.tight_layout()
    return fig


def _reduce_rows(name: str, a: np.ndarray, axis: int) -> np.ndarray:
    a = _as_2d(name, a, mask_future=False)
    return np.sqrt((a ** 2).sum(axis=1 - axis))


def plot_evolution(run: Run, field: str, name: str, axis: int = 0, example: int = 0,
                   log: bool = False) -> plt.Figure:
    """Norm of each row (axis=0) or column (axis=1) of one tensor, over recorded steps.

    For activations, rows are sequence positions and columns hidden units.
    """
    cols = [_reduce_rows(name, get_field(s, field, example)[name], axis) for s in run.iter_snapshots()]
    mat = np.array(cols).T
    fig, ax = plt.subplots(figsize=(10, 5))
    norm = None
    if log:
        positive = mat[mat > 0]
        norm = LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.size else None
        mat = np.where(mat > 0, mat, np.nan)
    im = ax.imshow(mat, aspect="auto", cmap="magma", interpolation="nearest", norm=norm)
    xt = np.linspace(0, len(run.steps) - 1, min(10, len(run.steps))).astype(int)
    ax.set_xticks(xt, [run.steps[i] for i in xt], fontsize=7)
    ax.set_xlabel("step (one column per snapshot)")
    ax.set_ylabel("row" if axis == 0 else "column")
    plt.colorbar(im, ax=ax, label="norm")
    ax.set_title(f"{run.dir.name}: {field} / {name}, {'row' if axis == 0 else 'column'} norms", fontsize=10)
    fig.tight_layout()
    return fig


def animate(run: Run, field: str, name: str, out_path: str | Path, example: int = 0, fps: int = 8,
            fixed_scale: bool = False) -> None:
    """GIF of one tensor's heatmap over recorded steps.

    With fixed_scale, all frames share one colour scale (taken over all steps);
    otherwise each frame is scaled to its own range so structure stays visible.
    """
    frames = [(s["step"], get_field(s, field, example)[name]) for s in run.iter_snapshots()]
    vlim = max(_symmetric_limit(_as_2d(name, a)) for _, a in frames) if fixed_scale else None
    fig = plt.figure(figsize=(6, 5))

    def draw(i):
        fig.clear()
        ax = fig.add_subplot()
        step, a = frames[i]
        heatmap(ax, name, a, title=f"{field} / {name}\nstep {step}", vlim=vlim)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
