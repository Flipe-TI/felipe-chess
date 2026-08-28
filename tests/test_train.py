"""Testes do train.py: métricas top-k (puras), avaliação, early stopping e o
loop de treino end-to-end em dados minúsculos (aprende + salva checkpoint).
"""
import json

import numpy as np
import torch

from felipe_chess.model import PolicyNet
from felipe_chess.train import (
    EarlyStopper,
    evaluate,
    top_k_accuracy,
    train_model,
)


def test_top_k_accuracy_top1():
    # 2 exemplos: argmax certo no 1º, errado no 2º.
    logits = torch.tensor([[0.1, 5.0, 0.2], [9.0, 0.1, 0.2]])
    targets = torch.tensor([1, 2])
    assert top_k_accuracy(logits, targets, k=1) == 0.5


def test_top_k_accuracy_top3_catches_second_choice():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])  # alvo é o 2º melhor
    targets = torch.tensor([2])
    assert top_k_accuracy(logits, targets, k=1) == 0.0
    assert top_k_accuracy(logits, targets, k=3) == 1.0


def test_evaluate_returns_metrics():
    torch.manual_seed(0)
    net = PolicyNet(channels=8, blocks=1).eval()
    X = np.random.rand(6, 18, 8, 8).astype(np.float32)
    y = np.random.randint(0, 4672, size=6).astype(np.int32)
    m = evaluate(net, X, y, batch_size=4)
    assert set(m) >= {"top1", "top3", "loss"}
    assert 0.0 <= m["top1"] <= 1.0 and 0.0 <= m["top3"] <= 1.0
    assert m["top3"] >= m["top1"]


def test_early_stopper_triggers_after_patience():
    stop = EarlyStopper(patience=2)
    assert stop.update(0.10) is False  # melhora (baseline)
    assert stop.update(0.20) is False  # melhora -> reseta
    assert stop.update(0.19) is False  # piora 1
    assert stop.update(0.18) is True   # piora 2 -> para
    assert stop.best == 0.20


def _tiny_npz(path, n=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 18, 8, 8), dtype=np.float32)
    y = rng.integers(0, 4672, size=n).astype(np.int32)
    np.savez(path, X=X, y=y)


def test_train_model_learns_and_saves_checkpoint(tmp_path):
    train_p = tmp_path / "train.npz"
    hold_p = tmp_path / "holdout.npz"
    _tiny_npz(train_p, n=8, seed=1)
    _tiny_npz(hold_p, n=4, seed=2)

    result = train_model(
        train_path=train_p,
        holdout_path=hold_p,
        out_dir=tmp_path / "models",
        channels=16,
        blocks=1,
        epochs=60,
        lr=1e-2,
        weight_decay=0.0,
        patience=100,  # não parar cedo neste teste
        batch_size=8,
        seed=0,
    )
    assert (tmp_path / "models" / result["checkpoint"]).exists() or (
        tmp_path / "models"
    ).joinpath("policy.pt").exists()
    # Overfit dos 8 exemplos de treino: top-1 de treino chega alto.
    assert result["history"][-1]["train_top1"] >= 0.99
    assert len(result["history"]) <= 60

    # persiste metrics.json (o guard do retrain lê isso)
    metrics = json.loads((tmp_path / "models" / "metrics.json").read_text(encoding="utf-8"))
    assert "holdout_top1" in metrics and "holdout_top3" in metrics
    assert metrics["holdout_top1"] == result["best"]["holdout_top1"]
