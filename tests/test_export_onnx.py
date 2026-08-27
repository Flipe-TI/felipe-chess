"""Testes do export_onnx: ONNX válido, contrato de I/O (board[1,18,8,8] ->
policy[1,4672]) e parity numérica ONNX<->PyTorch. NÃO toca no checkpoint real
(models/policy.pt) — usa tmp_path + modelo fresco.
"""
import numpy as np
import onnx
import torch

from felipe_chess.model import PolicyNet
from felipe_chess.export_onnx import (
    export_onnx,
    load_model_from_checkpoint,
    verify_onnx_matches_torch,
)


def test_export_produces_valid_onnx(tmp_path):
    net = PolicyNet(channels=16, blocks=1)
    path = export_onnx(net, tmp_path / "m.onnx")
    assert path.exists()
    onnx.checker.check_model(onnx.load(str(path)))  # não levanta = válido


def test_onnx_io_contract(tmp_path):
    net = PolicyNet(channels=16, blocks=1)
    path = export_onnx(net, tmp_path / "m.onnx")
    m = onnx.load(str(path))

    inp = m.graph.input[0]
    out = m.graph.output[0]
    assert inp.name == "board"
    assert out.name == "policy"

    def shape(t):
        return [d.dim_value for d in t.type.tensor_type.shape.dim]

    assert shape(inp) == [1, 18, 8, 8]
    assert shape(out) == [1, 4672]


def test_onnx_matches_torch(tmp_path):
    torch.manual_seed(0)
    net = PolicyNet(channels=16, blocks=2)
    path = export_onnx(net, tmp_path / "m.onnx")
    max_diff = verify_onnx_matches_torch(net, path, n=8, tol=1e-4)
    assert max_diff < 1e-4, f"ONNX diverge do PyTorch: max_diff={max_diff}"


def test_load_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = PolicyNet(channels=16, blocks=1).eval()
    ckpt = tmp_path / "policy.pt"
    torch.save(
        {"state_dict": net.state_dict(), "channels": 16, "blocks": 1,
         "encoding_version": "az-8x8x73-v1"},
        ckpt,
    )
    loaded = load_model_from_checkpoint(ckpt).eval()
    x = torch.randn(1, 18, 8, 8)
    with torch.no_grad():
        assert torch.allclose(net(x), loaded(x), atol=1e-6)
