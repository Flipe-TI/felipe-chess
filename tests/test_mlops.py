"""Testes da lógica de MLOps: guard de publicação (só se melhorar) e a
construção do meta.json que vai pro site.
"""
import json

from felipe_chess.mlops import build_site_meta, should_publish


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
