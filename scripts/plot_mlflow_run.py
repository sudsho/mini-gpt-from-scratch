"""quick utility to plot train/val loss curves for a given mlflow run id.

usage:
    python scripts/plot_mlflow_run.py --run-id <id> --out loss_curve.png
"""

import argparse
import matplotlib.pyplot as plt
import mlflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="loss_curve.png")
    args = ap.parse_args()

    client = mlflow.tracking.MlflowClient()
    train = client.get_metric_history(args.run_id, "train_loss")
    val = client.get_metric_history(args.run_id, "val_loss")

    if not train or not val:
        print("no train_loss/val_loss metrics on run, did training log them?")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([m.step for m in train], [m.value for m in train], label="train")
    ax.plot([m.step for m in val], [m.value for m in val], label="val")
    ax.set_xlabel("iter")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
