"""Lógica de MLOps do retrain diário: guard de publicação e meta.json do site.

Mantido pequeno e testável; o workflow (.github/workflows/retrain.yml) é só a
cola em cima disto.
"""
from __future__ import annotations

import csv
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


# --- Histórico dos retrains (CSV append-only, versionado) ------------------
# Uma linha por retrain para acompanhar a evolução (abre no Power BI/Excel).
HISTORY_FIELDS = [
    "date",
    "n_games",
    "n_samples",
    "holdout_top1",
    "holdout_top3",
    "epochs",
    "baseline_top1",
    "published",
]


def append_history(csv_path, record: dict) -> None:
    """Acrescenta uma linha ao CSV de histórico (escreve o header se for novo)."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in HISTORY_FIELDS})


def append_run(
    history_csv,
    meta_path,
    metrics_path,
    published_metrics_path,
    published: bool,
    date: str | None = None,
) -> dict:
    """Monta o registro de UM retrain (a partir dos jsons) e o acrescenta ao CSV.

    `published_metrics_path` deve ser lido ANTES de ser sobrescrito pelo baseline
    novo, para registrar contra qual valor este run foi comparado.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    base = ""
    p = Path(published_metrics_path)
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8")).get("holdout_top1")
        base = round(float(raw), 4) if raw is not None else ""

    record = {
        "date": date or _dt.date.today().isoformat(),
        "n_games": meta.get("n_games_train", 0) + meta.get("n_games_holdout", 0),
        "n_samples": meta.get("n_total"),
        "holdout_top1": round(float(m["holdout_top1"]), 4),
        "holdout_top3": round(float(m["holdout_top3"]), 4),
        "epochs": m.get("epochs"),
        "baseline_top1": base,
        "published": bool(published),
    }
    append_history(history_csv, record)
    return record


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
