"""S1 — baixa as partidas do Felipe da API pública do Chess.com.

Fonte: https://api.chess.com/pub/player/{username}/games/archives → arquivos
mensais → cada um com os PGNs. Sem chave/auth. Salva um PGN por mês em
data/felipe/ (versionado, pequeno).

Idempotente: pula meses já baixados e SEMPRE re-baixa o mês mais recente
(pode ter jogos novos — é o loop de evolução do BRAINSTORM §O3).

O transporte HTTP é injetável (`fetch_json`) para testes offline.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ARCHIVES_URL = "https://api.chess.com/pub/player/{username}/games/archives"
USER_AGENT = "felipe-chess/0.1 (behavior-cloning dataset; contact felipegabriel.suporteti@gmail.com)"


def _default_fetch_json(url: str) -> dict:
    """GET url e decodifica JSON. User-Agent educado (a API bloqueia UA vazio)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def list_archives(username: str, fetch_json=_default_fetch_json) -> list[str]:
    """Lista as URLs de arquivos mensais (ordem cronológica, como a API devolve)."""
    data = fetch_json(ARCHIVES_URL.format(username=username))
    return list(data.get("archives", []))


def archive_url_to_month(url: str) -> str:
    """'.../games/2025/09' -> '2025-09'."""
    year, month = url.rstrip("/").split("/")[-2:]
    return f"{year}-{month}"


def filter_games(games, time_classes=("rapid",), rules: str = "chess") -> list[dict]:
    """Mantém só jogos de xadrez padrão nos time controls pedidos."""
    tc = set(time_classes)
    return [
        g for g in games
        if g.get("rules") == rules and g.get("time_class") in tc
    ]


def games_to_pgn(games) -> str:
    """Concatena os campos `pgn` num arquivo PGN (jogos separados por linha em branco)."""
    pgns = [g["pgn"].strip() for g in games if g.get("pgn")]
    if not pgns:
        return ""
    return "\n\n".join(pgns) + "\n"


def fetch_player_games(
    username: str,
    out_dir,
    time_classes=("rapid",),
    rules: str = "chess",
    fetch_json=_default_fetch_json,
    delay: float = 0.0,
) -> dict:
    """Baixa os jogos e escreve um PGN por mês em out_dir. Retorna um resumo.

    Meses já baixados são pulados, exceto o mais recente (sempre re-baixado).
    Meses sem jogos (após filtro) não geram arquivo.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    archives = list_archives(username, fetch_json=fetch_json)
    latest_url = archives[-1] if archives else None

    written: dict[str, int] = {}
    skipped: list[str] = []

    for url in archives:
        month = archive_url_to_month(url)
        path = out / f"{month}.pgn"
        is_latest = url == latest_url

        if path.exists() and not is_latest:
            skipped.append(month)
            continue

        if delay:
            time.sleep(delay)
        games = filter_games(
            fetch_json(url).get("games", []), time_classes=time_classes, rules=rules
        )
        pgn = games_to_pgn(games)
        if not pgn:
            # Mês vazio após filtro: não escreve arquivo.
            if path.exists() and is_latest:
                # Havia arquivo antigo mas o mês ficou vazio; mantém como está.
                pass
            continue
        path.write_text(pgn, encoding="utf-8")
        written[month] = len(games)

    return {"written": written, "skipped": skipped}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baixa as partidas do Felipe do Chess.com.")
    parser.add_argument("--username", default="felipoww")
    parser.add_argument("--out", default="data/felipe")
    parser.add_argument(
        "--time-classes", default="rapid",
        help="lista separada por vírgula (ex.: rapid,blitz). Default: rapid",
    )
    parser.add_argument("--delay", type=float, default=0.5, help="pausa entre requests (s)")
    args = parser.parse_args()

    tcs = tuple(t.strip() for t in args.time_classes.split(",") if t.strip())
    result = fetch_player_games(args.username, args.out, time_classes=tcs, delay=args.delay)
    total = sum(result["written"].values())
    print(f"Baixados {total} jogos em {len(result['written'])} meses -> {args.out}")
    for month, n in sorted(result["written"].items()):
        print(f"  {month}: {n}")
    if result["skipped"]:
        print(f"Pulados (já existiam): {', '.join(sorted(result['skipped']))}")
