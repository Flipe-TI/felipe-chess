"""Chess Position Encoder — az-8x8x73-v1 (port Python do encoding.mjs do site).

Fonte única da verdade: ../Flipe-TI.github.io/training/ENCODING.md
Este módulo DEVE reproduzir bit a bit o encoding.mjs. O parity test
(tests/test_encoding_parity.py) garante isso contra a fixture do site.

Sem dependências além do numpy; o FEN é parseado na mão, igual ao JS.
"""
from __future__ import annotations

import numpy as np

ENCODING_VERSION = "az-8x8x73-v1"
PLANES = 18
INPUT_SHAPE = (PLANES, 8, 8)
POLICY_SIZE = 4672  # 64 * 73

# --- Move encoding — AlphaZero 8x8x73 -------------------------------------
# Direções de dama, ordem congelada d=0..7: (file_delta, rank_delta)
QUEEN_DIRS = (
    (0, 1),    # 0 N
    (1, 1),    # 1 NE
    (1, 0),    # 2 E
    (1, -1),   # 3 SE
    (0, -1),   # 4 S
    (-1, -1),  # 5 SW
    (-1, 0),   # 6 W
    (-1, 1),   # 7 NW
)
# Offsets de cavalo, ordem congelada j=0..7 (moveType 56..63): (file_delta, rank_delta)
KNIGHT_OFFSETS = (
    (1, 2),    # 56
    (2, 1),    # 57
    (2, -1),   # 58
    (1, -2),   # 59
    (-1, -2),  # 60
    (-2, -1),  # 61
    (-2, 1),   # 62
    (-1, 2),   # 63
)
UNDER_PIECE = {"n": 0, "b": 1, "r": 2}  # underpromotion → índice
UNDER_PIECE_INV = ("n", "b", "r")

# Tipo de peça → offset de plano dentro do bloco "minhas peças" (P=0..K=5)
PIECE_PLANE = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def _parse_square(sq: str) -> tuple[int, int]:
    """'e2' -> (file=4, rank=1), 0-indexed."""
    return ord(sq[0]) - 97, int(sq[1]) - 1


def _format_square(file: int, rank: int) -> str:
    """(file, rank) -> 'e2'."""
    return chr(97 + file) + str(rank + 1)


def to_perspective_square(square: str, side_to_move: str) -> str:
    """Converte um square para a perspectiva do lado a mover.

    Brancas ('w'): inalterado. Pretas ('b'): rank-flip (r -> 9-r em notação),
    file inalterado. Ex.: 'd7' -> 'd2'. Mesma implementação canônica do JS.
    """
    if side_to_move != "b":
        return square
    mirrored_rank = 9 - int(square[1])
    return square[0] + str(mirrored_rank)


def move_to_index(from_square: str, to_square: str, promotion: str | None = None) -> int:
    """Codifica um lance para índice de policy AlphaZero [0, POLICY_SIZE).

    Squares já em perspectiva do lado a mover (o caller espelha antes).
    promotion: 'q'|'r'|'b'|'n'|None. 'q' e None usam o range de queen-move.
    """
    ff, fr = _parse_square(from_square)
    tf, tr = _parse_square(to_square)
    df = tf - ff
    dr = tr - fr
    from_sq = fr * 8 + ff

    # --- Underpromotion (n/b/r apenas) ---
    if promotion in ("n", "b", "r"):
        direction = df + 1  # file_delta -1->0, 0->1, +1->2
        piece = UNDER_PIECE[promotion]
        move_type = 64 + direction * 3 + piece
        return from_sq * 73 + move_type

    # --- Cavalo ---
    for j, (kdf, kdr) in enumerate(KNIGHT_OFFSETS):
        if kdf == df and kdr == dr:
            return from_sq * 73 + (56 + j)

    # --- Queen move (inclui promoção a dama; promotion 'q' ou None) ---
    k = max(abs(df), abs(dr))  # distância 1..7
    sdf = 0 if df == 0 else (1 if df > 0 else -1)
    sdr = 0 if dr == 0 else (1 if dr > 0 else -1)
    for d, (qdf, qdr) in enumerate(QUEEN_DIRS):
        if qdf == sdf and qdr == sdr:
            move_type = d * 7 + (k - 1)
            return from_sq * 73 + move_type

    raise ValueError(
        f"move_to_index: não consigo codificar {from_square}-{to_square} promo={promotion}"
    )


def index_to_move(index: int) -> tuple[str, str, str | None]:
    """Decodifica índice -> (from, to, promotion). Espelha indexToMove do JS.

    ATENÇÃO: a inferência de promoção é lossy/heurística (ver ENCODING.md);
    NÃO usar para decodificar um argmax de policy sem máscara de legalidade.
    Aqui serve só para round-trip nos testes.
    """
    from_sq = index // 73
    move_type = index % 73
    from_rank = from_sq // 8
    from_file = from_sq % 8

    # --- Underpromotion ---
    if move_type >= 64:
        sub = move_type - 64
        direction = sub // 3  # 0=left, 1=push, 2=right
        piece = sub % 3
        df = direction - 1
        dr = 1
        return (
            _format_square(from_file, from_rank),
            _format_square(from_file + df, from_rank + dr),
            UNDER_PIECE_INV[piece],
        )

    # --- Cavalo ---
    if move_type >= 56:
        j = move_type - 56
        kdf, kdr = KNIGHT_OFFSETS[j]
        return (
            _format_square(from_file, from_rank),
            _format_square(from_file + kdf, from_rank + kdr),
            None,
        )

    # --- Queen move ---
    d = move_type // 7
    k = (move_type % 7) + 1
    qdf, qdr = QUEEN_DIRS[d]
    to_file = from_file + qdf * k
    to_rank = from_rank + qdr * k

    promotion = None
    if k == 1 and qdr == 1 and to_rank == 7:
        promotion = "q"

    return (
        _format_square(from_file, from_rank),
        _format_square(to_file, to_rank),
        promotion,
    )


def encode_position(fen: str) -> np.ndarray:
    """Parseia um FEN e retorna tensor float32 [18, 8, 8].

    Índice flat: plane*64 + rank*8 + file (rank 0 = 1ª fileira das brancas).
    Sempre na perspectiva do lado a mover:
      - brancas: sem transform.
      - pretas: rank-flip (r -> 7-r) + color-swap (minhas peças nos planos 0-5).
    """
    parts = fen.strip().split()
    piece_placement = parts[0]
    active_color = parts[1]
    castling = parts[2] if len(parts) > 2 else "-"
    en_passant = parts[3] if len(parts) > 3 else "-"

    black_to_move = active_color == "b"
    tensor = np.zeros((PLANES, 8, 8), dtype=np.float32)

    # --- Peças ---
    # FEN lista fileiras de rank 8 até rank 1 (topo -> baixo).
    # FEN row i (0 = rank 8) -> rank interno = 7 - i (antes de qualquer flip).
    fen_rows = piece_placement.split("/")
    for fen_row in range(8):
        internal_rank_white = 7 - fen_row
        file = 0
        for ch in fen_rows[fen_row]:
            if ch.isdigit():
                file += int(ch)
                continue
            piece_color = "w" if ch.isupper() else "b"
            piece_type = ch.lower()
            plane_offset = PIECE_PLANE.get(piece_type)
            if plane_offset is not None:
                if not black_to_move:
                    rank = internal_rank_white
                    plane = (0 if piece_color == "w" else 6) + plane_offset
                else:
                    rank = 7 - internal_rank_white
                    plane = (0 if piece_color == "b" else 6) + plane_offset
                tensor[plane, rank, file] = 1.0
            file += 1

    # --- Plano 12: side-to-move (constante 1.0) ---
    tensor[12, :, :] = 1.0

    # --- Planos 13-16: roque (relativo ao lado a mover) ---
    if not black_to_move:
        if "K" in castling:
            tensor[13, :, :] = 1.0
        if "Q" in castling:
            tensor[14, :, :] = 1.0
        if "k" in castling:
            tensor[15, :, :] = 1.0
        if "q" in castling:
            tensor[16, :, :] = 1.0
    else:
        if "k" in castling:
            tensor[13, :, :] = 1.0
        if "q" in castling:
            tensor[14, :, :] = 1.0
        if "K" in castling:
            tensor[15, :, :] = 1.0
        if "Q" in castling:
            tensor[16, :, :] = 1.0

    # --- Plano 17: en-passant ---
    if en_passant != "-":
        ep_file = ord(en_passant[0]) - 97
        ep_rank = int(en_passant[1]) - 1  # 0-indexed, perspectiva branca
        if black_to_move:
            ep_rank = 7 - ep_rank
        tensor[17, ep_rank, ep_file] = 1.0

    return tensor
