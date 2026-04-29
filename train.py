import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_preprocessing import WifiLTSChannelDataset


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int = 94) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_results_dir(save_dir: str) -> str:
    save_dir = os.path.abspath(save_dir)
    return save_dir if os.path.basename(save_dir) == "results" else os.path.join(save_dir, "results")


# ============================================================
# Model
# ============================================================

class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_shape=(2, 64, 2),
        output_shape=(52, 2),
        hidden_dims=(128, 128, 128),
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = int(np.prod(input_shape))
        output_dim = int(np.prod(output_shape))

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.output_shape = output_shape
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.net(x)
        return x.reshape(x.size(0), *self.output_shape)


# ============================================================
# Data
# ============================================================

def build_dataloaders(
    train_csv_path: str,
    eval_csv_path: str,
    batch_size: int = 64,
):
    train_dataset = WifiLTSChannelDataset(train_csv_path)
    eval_dataset = WifiLTSChannelDataset(eval_csv_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, eval_loader


# ============================================================
# I/Q-domain metrics
# ============================================================

def _iq_power(x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 0 and x.size(-1) == 2:
        return torch.sum(x ** 2, dim=-1)
    return x ** 2


def _iq_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_iq_power(x).clamp_min(0.0))


def rmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.mean(_iq_power(pred - target)) + eps)


# ============================================================
# Train / Evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()

    total_sq_error = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = rmse_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            sq_error = _iq_power(pred - y)
            total_sq_error += sq_error.sum().item()
            total_count += sq_error.numel()

    return float(np.sqrt(total_sq_error / max(total_count, 1)))


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()

    total_sq_error = 0.0
    total_count = 0

    total_abs_error = 0.0
    total_target_abs = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            diff = pred - y

            sq_error = _iq_power(diff)

            total_sq_error += sq_error.sum().item()
            total_count += sq_error.numel()

            total_abs_error += _iq_abs(diff).sum().item()
            total_target_abs += _iq_abs(y).sum().item()

    eval_rmse = float(np.sqrt(total_sq_error / max(total_count, 1)))
    eval_nmae = float(total_abs_error / max(total_target_abs, 1e-12))

    return eval_rmse, eval_nmae


# ============================================================
# Plot
# ============================================================

def _plot_metric(ax, epochs, values, title: str, ylabel: str, label: str, best_epoch):
    values = np.asarray(values, dtype=float)

    ax.plot(epochs, values, label=label)

    if best_epoch is not None and 1 <= best_epoch <= len(values):
        ax.scatter(
            [best_epoch],
            [values[best_epoch - 1]],
            color="red",
            marker="o",
            s=15,
            zorder=5,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both")
    ax.legend()


def save_train_plot(
    history: dict,
    save_path: str,
    best_epoch: int | None,
    best_eval_nmae: float,
) -> None:
    epochs = np.arange(1, len(history["train_rmse"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)

    _plot_metric(
        axes[0],
        epochs,
        history["train_rmse"],
        title="Train RMSE",
        ylabel="RMSE",
        label="Train RMSE",
        best_epoch=best_epoch,
    )

    _plot_metric(
        axes[1],
        epochs,
        history["eval_rmse"],
        title="Evaluation RMSE",
        ylabel="RMSE",
        label="Eval RMSE",
        best_epoch=best_epoch,
    )

    _plot_metric(
        axes[2],
        epochs,
        history["eval_nmae"],
        title="Evaluation NMAE",
        ylabel="NMAE",
        label="Eval NMAE",
        best_epoch=best_epoch,
    )

    axes[2].set_xlabel("Epoch")

    if best_epoch is None:
        summary = f"Best Eval NMAE: {best_eval_nmae:.6f}"
    else:
        summary = f"Best Eval NMAE: {best_eval_nmae:.6f} at epoch {best_epoch}"

    fig.text(0.5, 0.01, summary, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main training function
# ============================================================

def train_one_snr(
    snr_db: int,
    data_dir: str = "/home/jinx/project/CE01/data_set",
    save_dir: str = "/home/jinx/project/CE01/results",
    seed: int = 94,
    batch_size: int = 64,
    num_epochs: int = 200,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 1e-6,
    hidden_dims=(128, 128, 128),
    dropout: float = 0.1,
):
    set_seed(seed)

    results_dir = resolve_results_dir(save_dir)
    os.makedirs(results_dir, exist_ok=True)

    train_csv_path = os.path.join(data_dir, f"wifi_lltf_dataset_{snr_db}db.csv")
    eval_csv_path = os.path.join(data_dir, f"wifi_lltf_dataset_{snr_db}db_eval.csv")
    plot_path = os.path.join(results_dir, f"training_plot_{snr_db}db.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, eval_loader = build_dataloaders(
        train_csv_path=train_csv_path,
        eval_csv_path=eval_csv_path,
        batch_size=batch_size,
    )

    model = MLPRegressor(
        input_shape=(2, 64, 2),
        output_shape=(52, 2),
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_rmse": [],
        "eval_rmse": [],
        "eval_nmae": [],
    }

    best_eval_nmae = float("inf")
    best_eval_rmse = None
    best_epoch = None

    early_stop_best = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_rmse = train_one_epoch(model, train_loader, optimizer, device)
        eval_rmse, eval_nmae = evaluate(model, eval_loader, device)

        history["train_rmse"].append(train_rmse)
        history["eval_rmse"].append(eval_rmse)
        history["eval_nmae"].append(eval_nmae)

        if eval_nmae < best_eval_nmae:
            best_eval_nmae = eval_nmae
            best_eval_rmse = eval_rmse
            best_epoch = epoch

        print(
            f"SNR {snr_db:2d} dB | "
            f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
            f"Train RMSE: {train_rmse:.6f} | "
            f"Eval RMSE: {eval_rmse:.6f} | "
            f"Eval NMAE: {eval_nmae:.6f}"
        )

        improved_for_early_stop = eval_nmae < early_stop_best - early_stopping_min_delta

        if improved_for_early_stop:
            early_stop_best = eval_nmae
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best Eval NMAE: {best_eval_nmae:.6f}"
            )
            break

    save_train_plot(
        history=history,
        save_path=plot_path,
        best_epoch=best_epoch,
        best_eval_nmae=best_eval_nmae,
    )

    print("\nTraining finished.")
    print(f"Best Eval NMAE: {best_eval_nmae:.6f}")

    if best_eval_rmse is not None:
        print(f"Eval RMSE at best epoch: {best_eval_rmse:.6f}")

    if best_epoch is not None:
        print(f"Best epoch: {best_epoch}")

    print(f"Training plot saved to: {plot_path}")

    return {
        "snr_db": snr_db,
        "best_eval_nmae": best_eval_nmae,
        "best_eval_rmse": best_eval_rmse,
        "best_epoch": best_epoch,
        "plot_path": plot_path,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_db", type=int, default=18)
    parser.add_argument("--data_dir", type=str, default="/home/jinx/project/CE01/data_set")
    parser.add_argument("--save_dir", type=str, default="/home/jinx/project/CE01/results")
    args = parser.parse_args()

    train_one_snr(
        snr_db=args.snr_db,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )
