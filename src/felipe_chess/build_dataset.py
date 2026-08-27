"""S1 — transforma os PGNs do Felipe em amostras de treino (posição, lance).

Behavior cloning DELE: só emite amostra nas posições onde é a vez do
jogador-alvo, com alvo = o lance que ele jogou. Reusa encoding.py (posição
-> tensor 18×8×8; lance -> índice AlphaZero, já na perspectiva do lado a mover).

Saída: data/processed/{train,holdout}.npz + meta.json (com encoding_version,
exigido pelo ENCODING.md para rejeitar dados incompatíveis no treino).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from .encoding import (
    ENCODING_VERSION,
    encode_position,
    move_to_index,
    to_perspective_square,
)


def _move_index_in_perspective(board: chess.Board, move: chess.Move) -> int:
    """Índice AlphaZero do lance, na perspectiva do lado a mover (espelha pretas)."""
    side = "w" if board.turn == chess.WHITE else "b"
    uci = move.uci()  # ex.: "e2e4", "e7e8q"
    raw_from, raw_to = uci[0:2], uci[2:4]
    promotion = uci[4] if len(uci) > 4 else None
    from_sq = to_perspective_square(raw_from, side)
    to_sq = to_perspective_square(raw_to, side)
    return move_to_index(from_sq, to_sq, promotion)


def _samples_from_game(game: chess.pgn.Game, player: str) -> list[tuple[str, int]]:
    """Replaya o mainline; emite (fen, índice) só nos lances de `player`."""
    white = game.headers.get("White", "")
    black = game.headers.get("Black", "")
    if player == white:
        target = chess.WHITE
    elif player == black:
        target = chess.BLACK
    else:
        return []  # jogador não está nesta partida

    samples: list[tuple[str, int]] = []
    board = game.board()
    for move in game.mainline_moves():
        if board.turn == target:
            samples.append((board.fen(), _move_index_in_perspective(board, move)))
        board.push(move)
    return samples


def game_samples_from_pgn(pgn_text: str, player: str) -> list[tuple[str, int]]:
    """Amostras (fen, índice) do PRIMEIRO jogo no texto PGN. Útil pra testes."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return []
    return _samples_from_game(game, player)


def extract_samples(pgn_text: str, player: str) -> tuple[np.ndarray, np.ndarray]:
    """Percorre todos os jogos no texto PGN -> (X [N,18,8,8] f32, y [N] int32)."""
    stream = io.StringIO(pgn_text)
    fens: list[str] = []
    ys: list[int] = []
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        for fen, idx in _samples_from_game(game, player):
            fens.append(fen)
            ys.append(idx)

    if not fens:
        return (
            np.zeros((0, 18, 8, 8), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    X = np.stack([encode_position(f) for f in fens]).astype(np.float32)
    y = np.asarray(ys, dtype=np.int32)
    return X, y


def build_dataset(
    pgn_dir,
    out_dir,
    player: str = "felipoww",
    holdout_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Lê todos os *.pgn de pgn_dir, extrai amostras, embaralha (seed fixa) e
    grava train/holdout .npz + meta.json. Retorna um resumo."""
    pgn_dir = Path(pgn_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(pgn_dir.glob("*.pgn"))
    )
    X, y = extract_samples(combined, player)
    n = len(y)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_holdout = int(round(n * holdout_frac))
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]

    np.savez_compressed(out_dir / "train.npz", X=X[train_idx], y=y[train_idx])
    np.savez_compressed(out_dir / "holdout.npz", X=X[holdout_idx], y=y[holdout_idx])

    meta = {
        "encoding_version": ENCODING_VERSION,
        "player": player,
        "n_total": n,
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(holdout_idx)),
        "holdout_frac": holdout_frac,
        "seed": seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PGN -> amostras de treino (posição, lance).")
    parser.add_argument("--pgn-dir", default="data/felipe")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--player", default="felipoww")
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = build_dataset(
        args.pgn_dir, args.out, player=args.player,
        holdout_frac=args.holdout_frac, seed=args.seed,
    )
    print(json.dumps(meta, indent=2))
