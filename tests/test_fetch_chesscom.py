"""Testes offline do fetch_chesscom: transporte HTTP injetado, sem rede.
Cobrem parsing de arquivos, filtro (só rapid/chess), concat de PGN e
idempotência (pula meses já baixados, re-baixa o mais recente).
"""
from felipe_chess.fetch_chesscom import (
    archive_url_to_month,
    fetch_player_games,
    filter_games,
    games_to_pgn,
    list_archives,
)

ARCHIVES = "https://api.chess.com/pub/player/felipoww/games/archives"


def _game(pgn, time_class="rapid", rules="chess"):
    return {"pgn": pgn, "time_class": time_class, "rules": rules}


def make_transport(mapping):
    """Fake fetch_json: dict[url] -> objeto JSON já decodificado."""
    def fetch_json(url):
        return mapping[url]
    return fetch_json


def test_archive_url_to_month():
    assert archive_url_to_month("https://api.chess.com/pub/player/felipoww/games/2025/09") == "2025-09"
    assert archive_url_to_month("https://api.chess.com/pub/player/felipoww/games/2026/01/") == "2026-01"


def test_list_archives_returns_urls():
    t = make_transport({ARCHIVES: {"archives": ["u/2025/09", "u/2025/10"]}})
    assert list_archives("felipoww", fetch_json=t) == ["u/2025/09", "u/2025/10"]


def test_filter_keeps_only_rapid_chess():
    games = [
        _game("A", "rapid", "chess"),
        _game("B", "blitz", "chess"),
        _game("C", "rapid", "chess960"),
        _game("D", "bullet", "chess"),
    ]
    kept = filter_games(games, time_classes=("rapid",), rules="chess")
    assert [g["pgn"] for g in kept] == ["A"]


def test_filter_can_include_multiple_time_classes():
    games = [_game("A", "rapid"), _game("B", "blitz"), _game("C", "bullet")]
    kept = filter_games(games, time_classes=("rapid", "blitz"), rules="chess")
    assert [g["pgn"] for g in kept] == ["A", "B"]


def test_games_to_pgn_concatenates_with_blank_line():
    out = games_to_pgn([_game("[Event \"1\"]\n1. e4 e5"), _game("[Event \"2\"]\n1. d4 d5")])
    assert "[Event \"1\"]" in out and "[Event \"2\"]" in out
    assert "\n\n" in out  # jogos separados por linha em branco
    assert out.endswith("\n")


def test_games_to_pgn_empty():
    assert games_to_pgn([]) == ""


def _archive_map():
    return {
        ARCHIVES: {"archives": ["BASE/2025/09", "BASE/2025/10", "BASE/2025/11"]},
        "BASE/2025/09": {"games": [_game("SEP1"), _game("SEP2"), _game("SEPB", "blitz")]},
        "BASE/2025/10": {"games": [_game("OCT1")]},
        "BASE/2025/11": {"games": [_game("NOV1")]},  # mês mais recente
    }


def test_fetch_writes_one_pgn_per_nonempty_month(tmp_path):
    t = make_transport(_archive_map())
    summary = fetch_player_games("felipoww", tmp_path, fetch_json=t)
    assert (tmp_path / "2025-09.pgn").exists()
    assert (tmp_path / "2025-10.pgn").exists()
    assert (tmp_path / "2025-11.pgn").exists()
    sep = (tmp_path / "2025-09.pgn").read_text(encoding="utf-8")
    assert "SEP1" in sep and "SEP2" in sep
    assert "SEPB" not in sep  # blitz filtrado (default só rapid)
    assert summary["written"]["2025-09"] == 2


def test_fetch_skips_existing_but_refetches_latest(tmp_path):
    # Pré-existe um mês antigo e o mais recente, ambos com conteúdo "stale".
    (tmp_path / "2025-09.pgn").write_text("STALE-OLD\n", encoding="utf-8")
    (tmp_path / "2025-11.pgn").write_text("STALE-LATEST\n", encoding="utf-8")
    t = make_transport(_archive_map())
    summary = fetch_player_games("felipoww", tmp_path, fetch_json=t)

    # Mês antigo já existia -> pulado, conteúdo intacto.
    assert (tmp_path / "2025-09.pgn").read_text(encoding="utf-8") == "STALE-OLD\n"
    assert "2025-09" in summary["skipped"]

    # Mês mais recente -> sempre re-baixado, sobrescrito.
    latest = (tmp_path / "2025-11.pgn").read_text(encoding="utf-8")
    assert "STALE-LATEST" not in latest
    assert "NOV1" in latest
