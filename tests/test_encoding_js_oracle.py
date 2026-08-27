"""Parity cross-language DE VERDADE: exige que encoding.py reproduza o oráculo
gerado a partir do encoding.mjs do site (tools/gen_js_oracle.mjs).

Diferente de parity.json (fino: 0 promoções, 1 posição preta), o js_oracle
cobre TODOS os 64×73 branches de move, toPerspectiveSquare nas 64 casas ×
ambos os lados, e 12 posições com EP/roque em ambas as perspectivas.

Regenerar após mudança no encoding do site:  node tools/gen_js_oracle.mjs
"""
import json
from pathlib import Path

import numpy as np

from felipe_chess.encoding import (
    ENCODING_VERSION,
    encode_position,
    move_to_index,
    to_perspective_square,
)

ORACLE = json.loads(
    (Path(__file__).parent / "fixtures" / "js_oracle.json").read_text(encoding="utf-8")
)


def test_oracle_encoding_version():
    assert ORACLE["encoding_version"] == ENCODING_VERSION


def test_all_move_branches_match_js():
    mismatches = []
    for m in ORACLE["moves"]:
        got = move_to_index(m["from"], m["to"], m["promotion"])
        if got != m["index"]:
            mismatches.append((m["from"], m["to"], m["promotion"], m["index"], got))
    assert not mismatches, (
        f"{len(mismatches)}/{len(ORACLE['moves'])} moves divergem do JS. "
        f"Primeiros: {mismatches[:10]}"
    )


def test_to_perspective_square_matches_js():
    mismatches = []
    for p in ORACLE["perspective"]:
        got = to_perspective_square(p["square"], p["side"])
        if got != p["result"]:
            mismatches.append((p["square"], p["side"], p["result"], got))
    assert not mismatches, f"perspective diverge do JS: {mismatches[:10]}"


def test_all_positions_match_js():
    mismatches = []
    for pos in ORACLE["positions"]:
        nz = np.flatnonzero(encode_position(pos["fen"]).reshape(-1)).tolist()
        if nz != pos["nonzero_indices"]:
            mismatches.append(pos["fen"])
    assert not mismatches, f"posições divergem do JS: {mismatches}"
