from pathlib import Path

import torch
import torch.nn.functional as F

from forgetting_study.config import RecordConfig
from forgetting_study.data.datasets import IGNORE_INDEX
from forgetting_study.model import Transformer


def masked_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE_INDEX)


class Recorder:
    """Saves parameter gradients, activations and activation gradients to disk.

    Each snapshot `snapshots/step_XXXXXXX.pt` is a dict with
      step, epoch
      param_grads:      {"train_batch" | "probe_<kind>": {param_name: tensor}}
      activations:      {tap_name: tensor}             (on the probe batch)
      activation_grads: {kind: {tap_name: tensor}}     (on the probe batch)
      probe_loss:       {kind: float}
    where <kind> is a loss kind ("masked", and "all" for recall tasks).
    All quantities in a snapshot are taken at the same weights, before the
    optimizer step of that training step.
    """

    def __init__(self, run_dir: Path, model: Transformer, probe_x: torch.Tensor,
                 probe_y: dict[str, torch.Tensor], cfg: RecordConfig, total_steps: int):
        self.dir = Path(run_dir) / "snapshots"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.probe_x = probe_x
        self.probe_y = probe_y
        self.cfg = cfg
        self.total_steps = total_steps
        self.dtype = getattr(torch, cfg.dtype)
        torch.save(
            {"inputs": probe_x.cpu(), "targets": {k: v.cpu() for k, v in probe_y.items()}},
            Path(run_dir) / "probe.pt",
        )

    def due(self, step: int) -> bool:
        c = self.cfg
        early = step < c.early_until_step and step % c.early_every_steps == 0
        return early or step % c.every_steps == 0 or step == self.total_steps - 1

    def _store(self, t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(device="cpu", dtype=self.dtype)

    def record(self, step: int, epoch: int) -> None:
        """Call after loss.backward() on the training batch and before optimizer.step()."""
        params = dict(self.model.named_parameters())
        train_grads = {n: p.grad.detach().clone() for n, p in params.items() if p.grad is not None}
        snap = {
            "step": step,
            "epoch": epoch,
            "param_grads": {"train_batch": {n: self._store(g) for n, g in train_grads.items()}},
            "activations": None,
            "activation_grads": {},
            "probe_loss": {},
        }

        was_training = self.model.training
        self.model.eval()
        for kind, targets in self.probe_y.items():
            self.model.zero_grad(set_to_none=True)
            with self.model.recording() as taps:
                loss = masked_ce(self.model(self.probe_x), targets)
            loss.backward()
            if snap["activations"] is None:
                snap["activations"] = {k: self._store(v) for k, v in taps.items()}
            snap["activation_grads"][kind] = {k: self._store(v.grad) for k, v in taps.items() if v.grad is not None}
            snap["param_grads"][f"probe_{kind}"] = {
                n: self._store(p.grad) for n, p in params.items() if p.grad is not None
            }
            snap["probe_loss"][kind] = loss.item()
        self.model.train(was_training)

        for n, p in params.items():
            p.grad = train_grads.get(n)
        torch.save(snap, self.dir / f"step_{step:07d}.pt")
