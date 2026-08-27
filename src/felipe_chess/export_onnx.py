"""S2 — exporta o PolicyNet treinado para ONNX no contrato congelado do site.

Contrato (não mudar sem alinhar com o site):
  input  board  : float32 [1, 18, 8, 8]
  output policy : float32 [1, 4672]

Valida numericamente ONNX <-> PyTorch antes de publicar (parity). O site
consome exatamente este contrato via onnxruntime-web.

Uso: python -m felipe_chess.export_onnx --checkpoint models/policy.pt --out models/model.onnx
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from .model import INPUT_PLANES, PolicyNet

OPSET = 17
INPUT_NAME = "board"
OUTPUT_NAME = "policy"


def export_onnx(model: torch.nn.Module, out_path, opset: int = OPSET) -> Path:
    """Exporta o modelo para ONNX (batch fixo 1) e valida com onnx.checker."""
    model = model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, INPUT_PLANES, 8, 8)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # batch fixo 1, como o site espera
        dynamo=False,  # exportador legado (grafo shape-fixo, sem dep onnxscript)
    )
    onnx.checker.check_model(onnx.load(str(out_path)))
    return out_path


def verify_onnx_matches_torch(
    model: torch.nn.Module, onnx_path, n: int = 8, tol: float = 1e-4, seed: int = 0
) -> float:
    """Roda onnxruntime vs PyTorch em n entradas aleatórias; retorna o diff máx."""
    model = model.eval()
    sess = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    for _ in range(n):
        x = rng.random((1, INPUT_PLANES, 8, 8)).astype(np.float32)
        with torch.no_grad():
            t_out = model(torch.from_numpy(x)).numpy()
        o_out = sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
        max_diff = max(max_diff, float(np.abs(t_out - o_out).max()))
    return max_diff


def load_model_from_checkpoint(path) -> PolicyNet:
    """Reconstrói o PolicyNet a partir do checkpoint (channels/blocks/state_dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PolicyNet(channels=ckpt["channels"], blocks=ckpt["blocks"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exporta o PolicyNet treinado para ONNX.")
    parser.add_argument("--checkpoint", default="models/policy.pt")
    parser.add_argument("--out", default="models/model.onnx")
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.checkpoint)
    path = export_onnx(model, args.out)
    max_diff = verify_onnx_matches_torch(model, path, n=16, tol=args.tol)
    size_mb = path.stat().st_size / 1e6
    status = "OK" if max_diff < args.tol else "FALHOU"
    print(f"Exportado: {path} ({size_mb:.2f} MB)")
    print(f"Parity ONNX<->PyTorch: max_diff={max_diff:.2e} (tol={args.tol}) -> {status}")
