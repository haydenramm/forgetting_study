import json
import math
import time
from pathlib import Path

import torch
import yaml

from forgetting_study.config import RunConfig
from forgetting_study.data.datasets import IGNORE_INDEX, TaskData, load_task_data
from forgetting_study.model import Transformer
from forgetting_study.recorder import Recorder, masked_ce


@torch.no_grad()
def evaluate(model: Transformer, x: torch.Tensor, ys: dict[str, torch.Tensor], batch_size: int) -> dict:
    """Mean loss per loss kind, plus token accuracy on the masked targets."""
    model.eval()
    sums = {k: 0.0 for k in ys}
    counts = {k: 0 for k in ys}
    correct = 0
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size])
        for kind, y in ys.items():
            yb = y[i:i + batch_size]
            n = (yb != IGNORE_INDEX).sum().item()
            sums[kind] += masked_ce(logits, yb).item() * n
            counts[kind] += n
        yb = ys["masked"][i:i + batch_size]
        valid = yb != IGNORE_INDEX
        correct += ((logits.argmax(-1) == yb) & valid).sum().item()
    model.train()
    metrics = {f"test/loss_{k}": sums[k] / counts[k] for k in ys}
    metrics["test/acc"] = correct / counts["masked"]
    return metrics


def make_run_dir(cfg: RunConfig, config_stem: str) -> Path:
    name = cfg.name or f"{config_stem}-{cfg.task.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(cfg.out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train(cfg: RunConfig, run_dir: Path) -> dict:
    torch.manual_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    data: TaskData = load_task_data(cfg.task, device=device)
    model = Transformer(cfg.model, vocab_size=data.vocab_size, max_len=data.seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"task={data.name} vocab={data.vocab_size} seq_len={data.seq_len} "
          f"train={len(data.train_x)} test={len(data.test_x)} params={n_params:,}")

    tc = cfg.train
    steps_per_epoch = math.ceil(len(data.train_x) / tc.batch_size)
    total_steps = tc.epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    min_ratio = tc.min_lr / tc.lr
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * s / total_steps))
    )

    recorder = None
    if cfg.record.enabled:
        n = cfg.record.probe_size
        recorder = Recorder(run_dir, model, data.test_x[:n], {k: v[:n] for k, v in data.test_y.items()},
                            cfg.record, total_steps)

    gen = torch.Generator(device="cpu").manual_seed(tc.seed)
    metrics_file = open(run_dir / "metrics.jsonl", "w")
    step, metrics = 0, {}
    t0 = time.time()
    for epoch in range(tc.epochs):
        perm = torch.randperm(len(data.train_x), generator=gen).to(device)
        loss_sum = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * tc.batch_size:(i + 1) * tc.batch_size]
            loss = masked_ce(model(data.train_x[idx]), data.train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if recorder is not None and recorder.due(step):
                recorder.record(step, epoch)
            if tc.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            sched.step()
            loss_sum += loss.item()
            step += 1

        if (epoch + 1) % tc.eval_every_epochs == 0 or epoch == tc.epochs - 1:
            metrics = {"epoch": epoch + 1, "step": step, "lr": sched.get_last_lr()[0],
                       "train/loss": loss_sum / steps_per_epoch,
                       **evaluate(model, data.test_x, data.test_y, tc.batch_size)}
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            print(f"epoch {epoch + 1:4d} step {step:6d} train/loss {metrics['train/loss']:.4f} "
                  f"test/loss_masked {metrics['test/loss_masked']:.4f} test/acc {metrics['test/acc']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    metrics_file.close()
    torch.save(model.state_dict(), run_dir / "model.pt")
    return metrics
