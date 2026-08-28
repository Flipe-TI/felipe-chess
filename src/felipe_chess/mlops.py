"""Lógica de MLOps do retrain diário: guard de publicação e meta.json do site.

Mantido pequeno e testável; o workflow (.github/workflows/retrain.yml) é só a
cola em cima disto.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .encoding import ENCODING_VERSION
from .model import POLICY_SIZE

METRIC_KEY = "holdout_top1"


def _read_top1(path, default: float) -> float:
    p = Path(path)
    if not p.exists():
        return default
    return float(json.loads(p.read_text(encoding="utf-8")).get(METRIC_KEY, default))


def should_publish(new_metrics_path, published_metrics_path) -> bool:
    """Publica só se o modelo novo não for pior que o publicado (top-1 no holdout).

    Sem baseline publicado (arquivo ausente) → publica.
    """
    new = _read_top1(new_metrics_path, default=float("-inf"))
    published = _read_top1(published_metrics_path, default=float("-inf"))
    return new >= published


def build_site_meta(metrics: dict, encoding_version: str = ENCODING_VERSION) -> dict:
    """Monta o meta.json que acompanha o model.onnx no site (contrato + métricas)."""
    return {
        "encoding_version": encoding_version,
        "input_name": "board",
        "output_name": "policy",
        "input_shape": [1, 18, 8, 8],
        "policy_size": POLICY_SIZE,
        "fixture": False,
        "holdout_top1": metrics.get("holdout_top1"),
        "holdout_top3": metrics.get("holdout_top3"),
        "updated": _dt.date.today().isoformat(),
    }
