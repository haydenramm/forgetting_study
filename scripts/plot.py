"""Plot recorded heatmaps for a run. Figures are written to <run>/plots/.

Fields: param_grads/train_batch, param_grads/probe_masked, param_grads/probe_all,
        activations, activation_grads/masked, activation_grads/all

Usage:
    uv run scripts/plot.py summary   RUN                      # metrics, norms, snapshots at start/middle/end
    uv run scripts/plot.py metrics   RUN
    uv run scripts/plot.py snapshot  RUN FIELD [--step S] [--example I]
    uv run scripts/plot.py norms     RUN FIELD
    uv run scripts/plot.py evolution RUN FIELD TENSOR [--axis 0|1] [--example I] [--log]
    uv run scripts/plot.py animate   RUN FIELD TENSOR [--example I] [--fixed-scale]

Example:
    uv run scripts/plot.py animate runs/my-run activations block0.attn.probs --example 1
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from forgetting_study import plotting as P


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(path)


def slug(field: str) -> str:
    return field.replace("/", "-")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["summary", "metrics", "snapshot", "norms", "evolution", "animate"])
    parser.add_argument("run")
    parser.add_argument("field", nargs="?")
    parser.add_argument("tensor", nargs="?")
    parser.add_argument("--step", type=int, default=-1, help="recorded step (nearest is used); -1 = last")
    parser.add_argument("--example", type=int, default=0, help="probe example for activation fields")
    parser.add_argument("--axis", type=int, default=0, choices=[0, 1])
    parser.add_argument("--fixed-scale", action="store_true", help="animate: one colour scale for all frames")
    parser.add_argument("--log", action="store_true", help="evolution: logarithmic colour scale")
    args = parser.parse_args()

    run = P.Run(args.run)
    out = run.dir / "plots"
    needs = {"snapshot": ["field"], "norms": ["field"], "evolution": ["field", "tensor"], "animate": ["field", "tensor"]}
    for arg in needs.get(args.command, []):
        if getattr(args, arg) is None:
            parser.error(f"{args.command} requires {arg.upper()}")

    if args.command == "summary":
        save(P.plot_metrics(run), out / "metrics.png")
        middle = run.steps[len(run.steps) // 2]
        for field in run.fields():
            save(P.plot_norms(run, field), out / f"norms_{slug(field)}.png")
            for step in (run.steps[0], middle, run.steps[-1]):
                fig = P.plot_snapshot(run, field, step, args.example)
                save(fig, out / f"snapshot_{slug(field)}_step{step:07d}.png")
    elif args.command == "metrics":
        save(P.plot_metrics(run), out / "metrics.png")
    elif args.command == "snapshot":
        snap_step = run.snapshot(args.step)["step"]
        save(P.plot_snapshot(run, args.field, args.step, args.example),
             out / f"snapshot_{slug(args.field)}_step{snap_step:07d}.png")
    elif args.command == "norms":
        save(P.plot_norms(run, args.field), out / f"norms_{slug(args.field)}.png")
    elif args.command == "evolution":
        save(P.plot_evolution(run, args.field, args.tensor, args.axis, args.example, args.log),
             out / f"evolution_{slug(args.field)}_{args.tensor}_axis{args.axis}{'_log' if args.log else ''}.png")
    elif args.command == "animate":
        path = out / f"anim_{slug(args.field)}_{args.tensor}.gif"
        path.parent.mkdir(parents=True, exist_ok=True)
        P.animate(run, args.field, args.tensor, path, args.example, fixed_scale=args.fixed_scale)
        print(path)


if __name__ == "__main__":
    main()
