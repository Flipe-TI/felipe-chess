"""Parity test: garante que encoding.py (Python) reproduz EXATAMENTE o
encoding.mjs (JS) do repo do site. Divergência silenciosa aqui = modelo joga
lixo. Ver BRAINSTORM.md §5.

Espelha a lógica de assets/chess/tests/encoding-parity.test.mjs e
encoding-moves.test.mjs do repo irmão.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from felipe_chess.encoding import (
    ENCODING_VERSION,
    POLICY_SIZE,
    encode_position,
    index_to_move,
    move_to_index,
    to_perspective_square,
)

FIXTURE = Path(__file__).parent / "fixtures" / "parity.json"
SITE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "Flipe-TI.github.io"
    / "assets"
    / "chess"
    / "tests"
    / "fixtures"
    / "parity.json"
)

CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))


def _split_uci(uci):
    """'e7e8q' -> ('e7', 'e8', 'q'); 'e2e4' -> ('e2', 'e4', None)."""
    return uci[0:2], uci[2:4], (uci[4] if len(uci) > 4 else None)


def test_encoding_version_matches_contract():
    assert ENCODING_VERSION == "az-8x8x73-v1"


def test_policy_size():
    assert POLICY_SIZE == 4672


@pytest.mark.parametrize("case", CASES, ids=[c["fen"] for c in CASES])
def test_position_nonzero_indices(case):
    tensor = encode_position(case["fen"])
    assert tensor.shape == (18, 8, 8)
    assert tensor.dtype == np.float32
    nz = np.flatnonzero(tensor.reshape(-1)).tolist()
    assert nz == case["nonzero_indices"]


@pytest.mark.parametrize("case", CASES, ids=[c["fen"] for c in CASES])
def test_move_indices_in_perspective(case):
    side_to_move = case["fen"].split()[1]  # "w" | "b"
    for uci, expected_idx in case["moves"].items():
        raw_from, raw_to, promotion = _split_uci(uci)
        from_sq = to_perspective_square(raw_from, side_to_move)
        to_sq = to_perspective_square(raw_to, side_to_move)
        assert move_to_index(from_sq, to_sq, promotion) == expected_idx


# --- Round-trips portados de encoding-moves.test.mjs -----------------------
# A fixture só tem 1 posição preta e 0 underpromotions; estes cobrem os
# caminhos que ela não exercita (promo dama vs underpromo, knight).

def test_underpromotion_distinct_from_queen_promotion():
    knight = move_to_index("a7", "a8", "n")
    queen = move_to_index("a7", "a8", "q")
    assert knight != queen


@pytest.mark.parametrize(
    "from_sq,to_sq,promotion",
    [
        ("e2", "e4", None),   # push simples
        ("g1", "f3", None),   # cavalo
        ("a7", "a8", "q"),    # promoção a dama (queen-move range)
        ("a7", "a8", "n"),    # underpromotion a cavalo (push)
        ("a7", "b8", "r"),    # underpromotion a torre (captura-direita)
    ],
)
def test_move_round_trip(from_sq, to_sq, promotion):
    idx = move_to_index(from_sq, to_sq, promotion)
    assert 0 <= idx < POLICY_SIZE
    assert index_to_move(idx) == (from_sq, to_sq, promotion)


# --- Guard anti-drift do vendored fixture ----------------------------------

@pytest.mark.skipif(
    not SITE_FIXTURE.exists(),
    reason="repo irmão (site) não está ao lado; guard de drift skipado",
)
def test_vendored_fixture_matches_site():
    site = json.loads(SITE_FIXTURE.read_text(encoding="utf-8"))
    assert CASES == site, (
        "tests/fixtures/parity.json divergiu do site. Recopie:\n"
        "  cp ../Flipe-TI.github.io/assets/chess/tests/fixtures/parity.json "
        "tests/fixtures/parity.json"
    )
