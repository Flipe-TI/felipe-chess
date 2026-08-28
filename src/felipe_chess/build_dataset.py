"""S1 — transforma os PGNs do Felipe em amostras de treino (posição, lance).

Behavior cloning DELE: só emite amostra nas posições onde é a vez do
jogador-alvo, com alvo = o lance que ele jogou. Reusa encoding.py (posição
-> tensor 18×8×8; lance -> índice AlphaZero, já na perspectiva do lado a mover).

Saída: data/processed/{train,holdout}.npz + meta.json (com encoding_version,
exigido pelo ENCODING.md para rejeitar dados incompatíveis no treino).
"""
from __future__ import annotations

import hashlib
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


def game_key(game: chess.pgn.Game) -> str:
    """Identificador ESTÁVEL da partida (o `Link` do Chess.com é único).

    Fallback robusto quando não há Link. Usado para atribuir a partida a
    train/holdout de forma determinística e imutável conforme o dataset cresce.
    """
    h = game.headers
    link = h.get("Link")
    if link:
        return link
    return "|".join(
        h.get(k, "") for k in ("Site", "UTCDate", "UTCTime", "White", "Black")
    )


def is_holdout(key: str, holdout_frac: float, buckets: int = 100) -> bool:
    """Decide holdout por HASH da chave da partida (determinístico, cross-run).

    Usa md5 (não o `hash()` builtin, que é salgado por processo). Uma partida
    sempre cai no mesmo lado — sem vazamento game-level e sem reshuffle do
    holdout quando chegam partidas novas (o guard do retrain fica comparável).
    """
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % buckets
    return bucket < round(holdout_frac * buckets)


def iter_game_samples(pgn_text: str, player: str):
    """Itera as partidas do texto PGN -> (chave_estável, [(fen, índice), ...])."""
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        yield game_key(game), _samples_from_game(game, player)


def _encode_samples(fens: list[str], ys: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not fens:
        return (
            np.zeros((0, 18, 8, 8), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    X = np.stack([encode_position(f) for f in fens]).astype(np.float32)
    y = np.asarray(ys, dtype=np.int32)
    return X, y


def extract_samples(pgn_text: str, player: str) -> tuple[np.ndarray, np.ndarray]:
    """Percorre todos os jogos no texto PGN -> (X [N,18,8,8] f32, y [N] int32)."""
    fens: list[str] = []
    ys: list[int] = []
    for _key, samples in iter_game_samples(pgn_text, player):
        for fen, idx in samples:
            fens.append(fen)
            ys.append(idx)
    return _encode_samples(fens, ys)


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

    # Split POR PARTIDA (hash estável da chave): sem vazamento game-level e a
    # atribuição de cada jogo é imutável conforme o dataset cresce, então a
    # métrica do holdout é comparável entre retrains (o guard do retrain depende
    # disso). O `seed` é mantido no meta por reprodutibilidade/histórico.
    tr_fens: list[str] = []
    tr_ys: list[int] = []
    ho_fens: list[str] = []
    ho_ys: list[int] = []
    n_games_train = n_games_holdout = 0
    for key, samples in iter_game_samples(combined, player):
        if not samples:
            continue
        if is_holdout(key, holdout_frac):
            n_games_holdout += 1
            dst_fens, dst_ys = ho_fens, ho_ys
        else:
            n_games_train += 1
            dst_fens, dst_ys = tr_fens, tr_ys
        for fen, idx in samples:
            dst_fens.append(fen)
            dst_ys.append(idx)

    Xtr, ytr = _encode_samples(tr_fens, tr_ys)
    Xho, yho = _encode_samples(ho_fens, ho_ys)
    np.savez_compressed(out_dir / "train.npz", X=Xtr, y=ytr)
    np.savez_compressed(out_dir / "holdout.npz", X=Xho, y=yho)

    n = len(ytr) + len(yho)
    meta = {
        "encoding_version": ENCODING_VERSION,
        "player": player,
        "split": "by_game",
        "n_total": n,
        "n_train": int(len(ytr)),
        "n_holdout": int(len(yho)),
        "n_games_train": n_games_train,
        "n_games_holdout": n_games_holdout,
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
