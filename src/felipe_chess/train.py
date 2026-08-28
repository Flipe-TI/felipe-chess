"""S2 — treina o PolicyNet nos lances do Felipe (behavior cloning).

Métrica de sucesso = top-1/top-3 move-match no holdout (quão bem prevê o
lance que ELE jogaria), NÃO Elo. Regularização (weight-decay) + early stopping
porque são poucos dados (~12k) e o risco central é overfit.

Uso: python -m felipe_chess.train --data data/processed --out models
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .encoding import ENCODING_VERSION
from .model import PolicyNet


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Fração de exemplos cujo alvo está entre os top-k logits."""
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices  # [N, k]
    hit = (topk == targets.unsqueeze(1)).any(dim=1)
    return hit.float().mean().item()


class EarlyStopper:
    """Para quando a métrica (maior=melhor) não melhora por `patience` épocas."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("-inf")
        self.since_improved = 0

    def update(self, value: float) -> bool:
        if value > self.best:
            self.best = value
            self.since_improved = 0
            return False
        self.since_improved += 1
        return self.since_improved >= self.patience


def _load_npz(path):
    d = np.load(path)
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


@torch.no_grad()
def evaluate(model, X: np.ndarray, y: np.ndarray, batch_size: int = 512) -> dict:
    """top-1/top-3/loss médios sobre (X, y)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                       torch.from_numpy(y.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch_size)
    n = len(ds)
    tot_loss = tot1 = tot3 = 0.0
    for xb, yb in loader:
        logits = model(xb)
        tot_loss += loss_fn(logits, yb).item()
        tot1 += top_k_accuracy(logits, yb, 1) * len(yb)
        tot3 += top_k_accuracy(logits, yb, 3) * len(yb)
    return {"loss": tot_loss / n, "top1": tot1 / n, "top3": tot3 / n}


def train_model(
    train_path,
    holdout_path,
    out_dir,
    channels: int = 64,
    blocks: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    batch_size: int = 256,
    seed: int = 42,
    meta_path=None,
) -> dict:
    """Treina, avalia no holdout a cada época, early-stopping, salva o melhor
    checkpoint. Retorna {history, best, checkpoint}."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guard de encoding: rejeita dados de contrato diferente.
    if meta_path is not None and Path(meta_path).exists():
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        if meta.get("encoding_version") != ENCODING_VERSION:
            raise ValueError(
                f"encoding_version do dataset ({meta.get('encoding_version')}) "
                f"!= do treino ({ENCODING_VERSION})"
            )

    Xtr, ytr = _load_npz(train_path)
    Xho, yho = _load_npz(holdout_path)

    model = PolicyNet(channels=channels, blocks=blocks)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    stopper = EarlyStopper(patience=patience)
    checkpoint = "policy.pt"
    ckpt_path = out_dir / checkpoint
    history: list[dict] = []
    best_top1 = float("-inf")
    best_top3 = float("-inf")

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        tr = evaluate(model, Xtr, ytr, batch_size=batch_size)
        ho = evaluate(model, Xho, yho, batch_size=batch_size)
        history.append({
            "epoch": epoch,
            "train_top1": tr["top1"],
            "holdout_top1": ho["top1"],
            "holdout_top3": ho["top3"],
            "holdout_loss": ho["loss"],
        })

        if ho["top1"] > best_top1:
            best_top1 = ho["top1"]
            best_top3 = ho["top3"]
            torch.save(
                {"state_dict": model.state_dict(),
                 "channels": channels, "blocks": blocks,
                 "encoding_version": ENCODING_VERSION},
                ckpt_path,
            )

        if stopper.update(ho["top1"]):
            break

    metrics = {
        "holdout_top1": best_top1,
        "holdout_top3": best_top3,
        "epochs": len(history),
        "n_train": len(ytr),
        "n_holdout": len(yho),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "history": history,
        "best": {"holdout_top1": best_top1},
        "checkpoint": checkpoint,
        "metrics": metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treina o PolicyNet nos lances do Felipe.")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="models")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = Path(args.data)
    result = train_model(
        train_path=data / "train.npz",
        holdout_path=data / "holdout.npz",
        meta_path=data / "meta.json",
        out_dir=args.out,
        channels=args.channels, blocks=args.blocks,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, batch_size=args.batch_size, seed=args.seed,
    )
    h = result["history"][-1]
    print(f"Fim. epocas={len(result['history'])} | "
          f"melhor holdout top-1={result['best']['holdout_top1']:.3f} | "
          f"ultima: top1={h['holdout_top1']:.3f} top3={h['holdout_top3']:.3f}")
