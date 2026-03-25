"""
Extracción de jugadores de las cinco grandes ligas europeas.

Descarga datos vía **API-Football** (si hay `API_FOOTBALL_KEY` válida) o **TheSportsDB**
por defecto, normaliza columnas y guarda `data/raw_players.csv`.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

# Liga API-Football v3 → nombre legible (temporada de referencia).
API_FOOTBALL_BIG5: list[tuple[int, str]] = [
    (39, "Premier League"),
    (140, "La Liga"),
    (135, "Serie A"),
    (78, "Bundesliga"),
    (61, "Ligue 1"),
]

# Nombres de liga para TheSportsDB (`search_all_teams.php?l=...`).
THESPORTSDB_LEAGUES: list[str] = [
    "English Premier League",
    "Spanish La liga",
    "Italian Serie A",
    "German Bundesliga",
    "French Ligue 1",
]

DEFAULT_SEASON = 2023
THESPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json"
REQUEST_SLEEP_SEC = 0.35


def _api_football_session() -> tuple[requests.Session, str]:
    load_dotenv()
    key = (os.getenv("API_FOOTBALL_KEY") or "").strip()
    if not key or key == "tu_api_key_aqui":
        raise RuntimeError("API_FOOTBALL_KEY no configurada o es placeholder.")
    base = (os.getenv("APIFOOTBALL_BASE_URL") or "https://v3.football.api-sports.io").rstrip("/")
    s = requests.Session()
    s.headers.update({"x-apisports-key": key})
    return s, base


def _api_football_get(session: requests.Session, base: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
    r = session.get(f"{base}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _parse_height_cm(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).lower().replace(" ", "")
    if "cm" in s:
        try:
            return float(s.replace("cm", ""))
        except ValueError:
            return None
    return None


def _parse_weight_kg(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).lower().replace(" ", "")
    if "kg" in s:
        try:
            return float(s.replace("kg", ""))
        except ValueError:
            return None
    return None


def _row_api_football_squad(
    player: dict[str, Any],
    team: dict[str, Any],
    league_name: str,
) -> dict[str, Any]:
    return {
        "player_id": player.get("id"),
        "player_name": player.get("name") or "",
        "team_name": team.get("name") or "",
        "team_id": team.get("id"),
        "league_name": league_name,
        "nationality": "",
        "age": player.get("age"),
        "birth_date": None,
        "position": player.get("position") or "",
        "preferred_foot": "",
        "photo_url": player.get("photo") or "",
        "height_cm": None,
        "weight_kg": None,
        "shirt_number": player.get("number"),
        "goals_season": None,
        "assists_season": None,
        "minutes_season": None,
        "ha_ganado_champions": 0,
    }


def api_json_to_dataframe(api_payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convierte una lista de respuestas JSON de API-Football (cada una el cuerpo parseado)
    en un DataFrame con el esquema crudo unificado de jugador.

    Entiende estructuras con `response` → lista de bloques con `team`, `players` o
    `player` + `statistics`.
    """
    rows: list[dict[str, Any]] = []
    for payload in api_payloads:
        resp = payload.get("response", payload)
        if not isinstance(resp, list):
            if isinstance(resp, dict) and "players" in resp:
                team = resp.get("team") or {}
                league = resp.get("league") or {}
                ln = league.get("name", "") if isinstance(league, dict) else ""
                for p in resp.get("players") or []:
                    if isinstance(p, dict):
                        rows.append(_row_api_football_squad(p, team, ln))
            continue
        for block in resp:
            if not isinstance(block, dict):
                continue
            team = block.get("team") or {}
            league_name = ""
            lg = block.get("league")
            if isinstance(lg, dict):
                league_name = lg.get("name") or ""
            players = block.get("players")
            if isinstance(players, list):
                for p in players:
                    if isinstance(p, dict):
                        rows.append(_row_api_football_squad(p, team, league_name))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def thesportsdb_json_to_dataframe(payloads: list[dict[str, Any]], league_name: str) -> pd.DataFrame:
    """
    Convierte respuestas de `lookup_all_players.php` (lista bajo la clave `player`)
    al mismo esquema crudo que API-Football.
    """
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        plist = payload.get("player") if isinstance(payload, dict) else None
        if not isinstance(plist, list):
            continue
        for row in plist:
            if not isinstance(row, dict):
                continue
            dob = row.get("dateBorn") or ""
            age = None
            if isinstance(dob, str) and len(dob) >= 4 and dob[:4].isdigit():
                age = 2026 - int(dob[:4])
            photo = row.get("strCutout") or row.get("strThumb") or row.get("strRender") or ""
            h = _parse_height_cm(row.get("strHeight"))
            w = _parse_weight_kg(row.get("strWeight"))
            sn = row.get("strNumber")
            try:
                shirt = int(float(str(sn))) if sn not in (None, "") else None
            except ValueError:
                shirt = None
            rows.append(
                {
                    "player_id": row.get("idPlayer"),
                    "player_name": row.get("strPlayer") or "",
                    "team_name": row.get("strTeam") or "",
                    "team_id": row.get("idTeam"),
                    "league_name": league_name,
                    "nationality": row.get("strNationality") or "",
                    "age": age,
                    "birth_date": dob,
                    "position": row.get("strPosition") or "",
                    "preferred_foot": row.get("strSide") or "",
                    "photo_url": photo,
                    "height_cm": h,
                    "weight_kg": w,
                    "shirt_number": shirt,
                    "goals_season": None,
                    "assists_season": None,
                    "minutes_season": None,
                    "ha_ganado_champions": 0,
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_players_api_football(min_players: int = 80, season: int = DEFAULT_SEASON) -> pd.DataFrame:
    session, base = _api_football_session()
    rows: list[dict[str, Any]] = []

    for league_id, league_name in API_FOOTBALL_BIG5:
        if len(rows) >= min_players * 3:
            break
        time.sleep(REQUEST_SLEEP_SEC)
        data = _api_football_get(session, base, "/teams", {"league": league_id, "season": season})
        for entry in data.get("response") or []:
            if len(rows) >= min_players * 3:
                break
            team = entry.get("team") if isinstance(entry, dict) else None
            if not isinstance(team, dict) or team.get("id") is None:
                continue
            time.sleep(REQUEST_SLEEP_SEC)
            squad = _api_football_get(session, base, "/players/squads", {"team": team["id"]})
            for block in squad.get("response") or []:
                t = block.get("team") or team
                for p in block.get("players") or []:
                    if isinstance(p, dict):
                        rows.append(_row_api_football_squad(p, t, league_name))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"], keep="first")
    return df.head(max(min_players, min(len(df), 200))).reset_index(drop=True)


def _thesportsdb_key() -> str:
    load_dotenv()
    return (os.getenv("THESPORTSDB_API_KEY") or "3").strip()


def fetch_players_thesportsdb(min_players: int = 80) -> pd.DataFrame:
    key = _thesportsdb_key()
    base_url = f"{THESPORTSDB_BASE}/{key}"
    parts: list[pd.DataFrame] = []

    for league_name in THESPORTSDB_LEAGUES:
        time.sleep(REQUEST_SLEEP_SEC)
        r = requests.get(f"{base_url}/search_all_teams.php", params={"l": league_name}, timeout=60)
        r.raise_for_status()
        teams = r.json().get("teams") or []
        if not isinstance(teams, list):
            continue
        for team in teams:
            if not isinstance(team, dict) or not team.get("idTeam"):
                continue
            time.sleep(REQUEST_SLEEP_SEC)
            pr = requests.get(f"{base_url}/lookup_all_players.php", params={"id": team["idTeam"]}, timeout=60)
            pr.raise_for_status()
            df_part = thesportsdb_json_to_dataframe([pr.json()], league_name)
            if not df_part.empty:
                parts.append(df_part)
            if sum(len(p) for p in parts) >= min_players * 2:
                break
        if sum(len(p) for p in parts) >= min_players * 2:
            break

    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"], keep="first")
    return df.head(max(min_players, min(len(df), 200))).reset_index(drop=True)


def choose_provider() -> str:
    load_dotenv()
    pref = (os.getenv("DATA_PROVIDER") or "").strip().lower()
    if pref in ("api_football", "thesportsdb"):
        return pref
    key = (os.getenv("API_FOOTBALL_KEY") or "").strip()
    if key and key != "tu_api_key_aqui":
        return "api_football"
    return "thesportsdb"


def ingest(
    min_players: int = 80,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Orquesta la descarga según `DATA_PROVIDER` / claves y escribe CSV.
    """
    output_path = output_path or Path("data/raw_players.csv")
    provider = choose_provider()
    if provider == "api_football":
        try:
            df = fetch_players_api_football(min_players=min_players)
        except (RuntimeError, requests.RequestException):
            df = fetch_players_thesportsdb(min_players=min_players)
    else:
        df = fetch_players_thesportsdb(min_players=min_players)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Descargar jugadores big-5 → CSV.")
    parser.add_argument("--min-players", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("data/raw_players.csv"))
    args = parser.parse_args()
    df = ingest(min_players=args.min_players, output_path=args.output)
    print(f"Filas guardadas: {len(df)} → {args.output}")


if __name__ == "__main__":
    main()
