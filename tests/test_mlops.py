"""Testes da lógica de MLOps: guard de publicação (só se melhorar) e a
construção do meta.json que vai pro site.
"""
import json

from felipe_chess.mlops import (
    append_history,
    append_run,
    build_site_meta,
    should_publish,
)


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_should_publish_when_no_published_baseline(tmp_path):
    new = _write(tmp_path / "new.json", {"holdout_top1": 0.10})
    missing = tmp_path / "nao_existe.json"
    assert should_publish(new, missing) is True


def test_should_publish_when_improved(tmp_path):
    new = _write(tmp_path / "new.json", {"holdout_top1": 0.32})
    pub = _write(tmp_path / "pub.json", {"holdout_top1": 0.315})
    assert should_publish(new, pub) is True


def test_should_not_publish_when_worse(tmp_path):
    new = _write(tmp_path / "new.json", {"holdout_top1": 0.30})
    pub = _write(tmp_path / "pub.json", {"holdout_top1": 0.315})
    assert should_publish(new, pub) is False


def test_should_publish_when_equal(tmp_path):
    new = _write(tmp_path / "new.json", {"holdout_top1": 0.315})
    pub = _write(tmp_path / "pub.json", {"holdout_top1": 0.315})
    assert should_publish(new, pub) is True


def test_append_history_writes_header_once_and_rows(tmp_path):
    p = tmp_path / "history.csv"
    append_history(p, {"date": "2026-08-28", "holdout_top1": 0.31, "published": True})
    append_history(p, {"date": "2026-08-29", "holdout_top1": 0.33, "published": True})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3               # header + 2 linhas
    assert lines[0].startswith("date,")  # header uma vez só
    assert "2026-08-28" in lines[1]
    assert "2026-08-29" in lines[2]


def test_append_run_records_a_row_from_files(tmp_path):
    _write(tmp_path / "meta.json", {"n_total": 12076, "n_games_train": 356, "n_games_holdout": 30})
    _write(tmp_path / "metrics.json", {"holdout_top1": 0.3149, "holdout_top3": 0.4883, "epochs": 26})
    _write(tmp_path / "published.json", {"holdout_top1": 0.30})
    hist = tmp_path / "history.csv"

    rec = append_run(
        hist, tmp_path / "meta.json", tmp_path / "metrics.json",
        tmp_path / "published.json", published=True, date="2026-08-28",
    )
    assert rec["n_games"] == 386
    assert rec["n_samples"] == 12076
    assert rec["holdout_top1"] == 0.3149
    assert rec["baseline_top1"] == 0.30
    assert rec["published"] is True

    lines = hist.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # header + 1 run
    assert "386" in lines[1] and "12076" in lines[1]


def test_build_site_meta_has_contract_and_metrics():
    meta = build_site_meta({"holdout_top1": 0.315, "holdout_top3": 0.445})
    assert meta["encoding_version"] == "az-8x8x73-v1"
    assert meta["input_name"] == "board"
    assert meta["output_name"] == "policy"
    assert meta["input_shape"] == [1, 18, 8, 8]
    assert meta["policy_size"] == 4672
    assert meta["fixture"] is False
    assert meta["holdout_top1"] == 0.315
    assert meta["holdout_top3"] == 0.445
