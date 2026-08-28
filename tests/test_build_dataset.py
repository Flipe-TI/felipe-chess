"""Testes do build_dataset: PGN -> amostras (posição, índice de lance) só dos
lances do jogador-alvo. python-chess real, sem mock.
"""
import json

import numpy as np

from felipe_chess.encoding import encode_position, move_to_index
from felipe_chess.build_dataset import (
    build_dataset,
    extract_samples,
    game_samples_from_pgn,
    is_holdout,
)

PGN_WHITE = """[Event "t"]
[White "felipoww"]
[Black "opp"]

1. e4 e5 2. Nf3 Nc6 *
"""

PGN_BLACK = """[Event "t"]
[White "opp"]
[Black "felipoww"]

1. e4 e5 2. Nf3 Nc6 *
"""


def test_only_target_player_moves_as_white():
    samples = game_samples_from_pgn(PGN_WHITE, "felipoww")
    # felipoww joga de brancas: só os lances dele (e4, Nf3) = 2 amostras.
    assert len(samples) == 2
    fen0, y0 = samples[0]
    assert fen0.startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w")
    assert y0 == 877          # e2e4 (índice conhecido da fixture)
    assert y0 == move_to_index("e2", "e4", None)
    assert samples[1][1] == 501  # g1f3


def test_only_target_player_moves_as_black():
    samples = game_samples_from_pgn(PGN_BLACK, "felipoww")
    # felipoww de pretas: só 1...e5 e 2...Nc6 = 2 amostras.
    assert len(samples) == 2
    fen0, y0 = samples[0]
    assert " b " in fen0  # posição com pretas a mover
    # e7e5 espelhado p/ perspectiva das pretas = e2e4 -> mesmo índice 877.
    assert y0 == 877


def test_sample_tensor_matches_encoding():
    fen0, _ = game_samples_from_pgn(PGN_WHITE, "felipoww")[0]
    X = np.stack([encode_position(fen0)])
    assert X.shape == (1, 18, 8, 8)
    # a amostra guarda o FEN de antes do lance; o encoding bate com encoding.py
    assert np.array_equal(encode_position(fen0), X[0])


def test_extract_samples_sums_over_multiple_games():
    two = PGN_WHITE + "\n" + PGN_BLACK
    X, y = extract_samples(two, "felipoww")
    assert len(y) == 4
    assert X.shape == (4, 18, 8, 8)
    assert X.dtype == np.float32
    assert y.dtype == np.int32


def test_build_dataset_writes_npz_and_meta(tmp_path):
    pgn_dir = tmp_path / "felipe"
    pgn_dir.mkdir()
    (pgn_dir / "a.pgn").write_text(PGN_WHITE, encoding="utf-8")
    (pgn_dir / "b.pgn").write_text(PGN_BLACK, encoding="utf-8")
    out = tmp_path / "processed"

    summary = build_dataset(pgn_dir, out, player="felipoww", holdout_frac=0.5, seed=42)

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["encoding_version"] == "az-8x8x73-v1"
    assert meta["player"] == "felipoww"

    train = np.load(out / "train.npz")
    holdout = np.load(out / "holdout.npz")
    assert train["X"].shape[1:] == (18, 8, 8)
    assert train["X"].dtype == np.float32
    assert train["y"].dtype == np.int32
    # 4 amostras, split 50/50 -> 2 + 2, sem sobreposição total.
    assert len(train["y"]) + len(holdout["y"]) == 4
    assert summary["n_total"] == 4


PGN_GAME_A = """[Event "t"]
[Site "Chess.com"]
[White "felipoww"]
[Black "opp"]
[Link "https://www.chess.com/game/live/1"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 *
"""


def test_is_holdout_is_deterministic_and_key_based():
    # Pura e estável: mesma chave -> mesmo resultado, sempre (garante que a
    # atribuição de uma partida NÃO muda quando o dataset cresce).
    k = "https://www.chess.com/game/live/1"
    assert is_holdout(k, 0.1) == is_holdout(k, 0.1)
    # frac=1.0 -> tudo holdout; frac=0.0 -> nada holdout
    assert is_holdout(k, 1.0) is True
    assert is_holdout(k, 0.0) is False


def test_single_game_is_never_split(tmp_path):
    # Anti-vazamento: TODAS as amostras de uma partida vão pro mesmo lado.
    pgn_dir = tmp_path / "felipe"
    pgn_dir.mkdir()
    (pgn_dir / "a.pgn").write_text(PGN_GAME_A, encoding="utf-8")
    out = tmp_path / "processed"
    build_dataset(pgn_dir, out, player="felipoww", holdout_frac=0.5)

    n_train = len(np.load(out / "train.npz")["y"])
    n_hold = len(np.load(out / "holdout.npz")["y"])
    assert n_train + n_hold == 4               # 4 lances do felipoww
    assert (n_train == 0) != (n_hold == 0)     # exatamente UM lado vazio → não dividiu


def test_build_dataset_split_is_deterministic(tmp_path):
    pgn_dir = tmp_path / "felipe"
    pgn_dir.mkdir()
    (pgn_dir / "a.pgn").write_text(PGN_WHITE + "\n" + PGN_BLACK, encoding="utf-8")
    out1 = tmp_path / "p1"
    out2 = tmp_path / "p2"
    build_dataset(pgn_dir, out1, player="felipoww", holdout_frac=0.5, seed=7)
    build_dataset(pgn_dir, out2, player="felipoww", holdout_frac=0.5, seed=7)
    y1 = np.load(out1 / "train.npz")["y"]
    y2 = np.load(out2 / "train.npz")["y"]
    assert np.array_equal(y1, y2)
