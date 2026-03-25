"""
Ingeniería de características: del CSV crudo a **40 columnas binarias** (0 o 1).

`BINARY_FEATURE_NAMES` fija el orden de columnas; el árbol y Streamlit deben usar el mismo orden.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

BINARY_FEATURE_NAMES: list[str] = [
    "es_portero",
    "es_defensa",
    "es_centrocampista",
    "es_delantero",
    "juega_en_premier_league",
    "juega_en_laliga",
    "juega_en_serie_a",
    "juega_en_bundesliga",
    "juega_en_ligue1",
    "es_zurdo",
    "es_diestro",
    "tiene_menos_de_25",
    "entre_25_y_30",
    "tiene_mas_de_30",
    "es_goleador",
    "tiene_muchas_asistencias",
    "es_joven_promesa",
    "nacionalidad_europea",
    "nacionalidad_sudamericana",
    "nacionalidad_centroamerica_caribe",
    "nacionalidad_norteamerica",
    "nacionalidad_africana",
    "nacionalidad_asia_u_oceania",
    "altura_mayor_185",
    "peso_mayor_82",
    "ha_ganado_champions",
    "nacido_en_pais_big5",
    "juega_en_equipo_top_uefa",
    "dorsal_menor_que_12",
    "dorsal_mayor_que_25",
    "rol_ofensivo_principal",
    "rol_defensivo_principal",
    "liga_norte_centro_europa",
    "liga_sur_europa",
    "veterano_33_o_mas",
    "marcador_10_goles_o_mas",
    "asistente_5_o_mas",
    "pie_marcado_como_ambos",
    "tiene_imagen_api",
    "tiene_dorsal_registrado",
]

assert len(BINARY_FEATURE_NAMES) == 40

EUROPE = {
    "albania", "andorra", "armenia", "austria", "azerbaijan", "belarus", "belgium", "bosnia",
    "bulgaria", "croatia", "cyprus", "czech republic", "denmark", "england", "estonia",
    "faroe islands", "finland", "france", "georgia", "germany", "gibraltar", "greece",
    "hungary", "iceland", "ireland", "israel", "italy", "kazakhstan", "kosovo", "latvia",
    "liechtenstein", "lithuania", "luxembourg", "malta", "moldova", "monaco", "montenegro",
    "netherlands", "north macedonia", "northern ireland", "norway", "poland", "portugal",
    "romania", "russia", "san marino", "scotland", "serbia", "slovakia", "slovenia", "spain",
    "sweden", "switzerland", "turkey", "ukraine", "united kingdom", "wales", "czechia",
    "bosnia and herzegovina", "russian federation", "uk",
}

SOUTH_AM = {
    "argentina", "bolivia", "brazil", "chile", "colombia", "ecuador", "guyana", "paraguay",
    "peru", "suriname", "uruguay", "venezuela",
}

CENTRAL_CARIB = {
    "antigua", "bahamas", "barbados", "belize", "costa rica", "cuba", "dominica",
    "dominican republic", "el salvador", "grenada", "guatemala", "haiti", "honduras",
    "jamaica", "mexico", "nicaragua", "panama", "puerto rico", "martinique", "guadeloupe",
}

NORTH_AM = {"canada", "united states", "usa", "u.s.a.", "greenland"}

AFRICA = {
    "algeria", "angola", "benin", "botswana", "burkina faso", "burundi", "cameroon",
    "senegal", "nigeria", "morocco", "egypt", "ghana", "ivory coast", "côte d'ivoire",
    "south africa", "tunisia", "kenya", "uganda", "zambia", "zimbabwe",
}

ASIA_OCEANIA = {
    "afghanistan", "australia", "bahrain", "bangladesh", "china", "india", "indonesia",
    "iran", "iraq", "japan", "jordan", "korea republic", "south korea", "north korea",
    "malaysia", "new zealand", "pakistan", "philippines", "qatar", "saudi arabia",
    "singapore", "thailand", "uae", "united arab emirates", "vietnam", "uzbekistan",
}

BIG5_NAT = {"england", "spain", "italy", "germany", "france"}

TOP_TEAM_HINTS = frozenset(
    """
    manchester city manchester united liverpool chelsea arsenal tottenham
    real madrid barcelona atletico sevilla
    bayern dortmund leipzig leverkusen
    juventus inter milan ac milan napoli roma lazio
    psg marseille lyon monaco
    porto benfica ajax
    """.split()
)


def _norm_country(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _country_bucket(nationality: str) -> str | None:
    n = _norm_country(nationality)
    if not n:
        return None
    if n in EUROPE or "england" in n or "scotland" in n or "wales" in n:
        return "eu"
    if n in SOUTH_AM:
        return "sa"
    if n in CENTRAL_CARIB:
        return "cc"
    if n in NORTH_AM:
        return "na"
    if n in AFRICA:
        return "af"
    if n in ASIA_OCEANIA or "korea" in n:
        return "ao"
    return "other"


def _position_flags(pos: str) -> tuple[bool, bool, bool, bool, bool, bool]:
    p = (pos or "").lower()
    gk = any(x in p for x in ("goalkeeper", "keeper", "gk", "portero"))
    de = any(x in p for x in ("defender", "defence", "defense", "back", "defensa", "cb"))
    mid = any(x in p for x in ("midfield", "mediocampista", "mid"))
    fw = any(x in p for x in ("forward", "attacker", "winger", "striker", "delantero", "attack"))
    offensive = fw or "wing" in p
    defensive = gk or de
    return gk, de, mid, fw, offensive, defensive


def _league_flags(league_name: str) -> tuple[int, int, int, int, int, int, int]:
    L = (league_name or "").lower()
    premier = "premier" in L and "scottish" not in L
    laliga = "laliga" in L.replace(" ", "") or "la liga" in L
    serie = "serie a" in L or ("italian" in L and "serie" in L)
    bundes = "bundesliga" in L
    ligue = "ligue 1" in L or L.strip() == "ligue 1" or ("french" in L and "ligue" in L)
    north = premier or bundes
    south = laliga or serie
    return int(premier), int(laliga), int(serie), int(bundes), int(ligue), int(north), int(south)


def _top_team(team_name: str) -> int:
    t = (team_name or "").lower()
    return int(any(h in t for h in TOP_TEAM_HINTS))


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construye:
    - `X`: DataFrame solo con las 40 columnas en `BINARY_FEATURE_NAMES`, valores 0/1, NaN→0.
    - `y`: etiqueta de clase por jugador (string único).
    - `meta`: `player_id`, `player_name`, `photo_url` alineados fila a fila con `X`.
    """
    if df.empty:
        raise ValueError("El DataFrame de entrada está vacío.")

    work = df.copy()
    work["player_name"] = work["player_name"].astype(str).fillna("")
    work["player_id_str"] = work["player_id"].astype(str)

    data: dict[str, list[int]] = {c: [] for c in BINARY_FEATURE_NAMES}
    names: list[str] = []
    photos: list[str] = []
    pids: list[str] = []

    for _, row in work.iterrows():
        gk, de, mid, fw, off_r, def_r = _position_flags(str(row.get("position") or ""))
        prem, lal, ser, bun, lig1, liga_norte, liga_sur = _league_flags(str(row.get("league_name") or ""))

        foot = str(row.get("preferred_foot") or "").strip().lower()
        zurdo = int("left" in foot or "izquier" in foot)
        diestro = int("right" in foot or "derech" in foot)
        ambos = int("both" in foot or "amb" in foot)

        age = row.get("age")
        try:
            af = float(age) if age is not None and not (isinstance(age, float) and np.isnan(age)) else None
        except (TypeError, ValueError):
            af = None

        goals = row.get("goals_season")
        ast = row.get("assists_season")
        try:
            g = int(float(goals)) if goals is not None and str(goals) != "nan" else 0
        except (TypeError, ValueError):
            g = 0
        try:
            a = int(float(ast)) if ast is not None and str(ast) != "nan" else 0
        except (TypeError, ValueError):
            a = 0

        ucl = row.get("ha_ganado_champions")
        try:
            ucl_i = int(float(ucl)) if ucl is not None and str(ucl) != "nan" else 0
        except (TypeError, ValueError):
            ucl_i = 0

        h = row.get("height_cm")
        w = row.get("weight_kg")
        try:
            hf = float(h) if h is not None and str(h) != "nan" else None
        except (TypeError, ValueError):
            hf = None
        try:
            wf = float(w) if w is not None and str(w) != "nan" else None
        except (TypeError, ValueError):
            wf = None

        sh = row.get("shirt_number")
        try:
            sh_i = int(float(sh)) if sh is not None and str(sh) != "nan" else None
        except (TypeError, ValueError):
            sh_i = None

        nat = str(row.get("nationality") or "")
        b = _country_bucket(nat)
        nn = _norm_country(nat)

        photo = str(row.get("photo_url") or "").strip()
        tiene_img = int(bool(photo) and photo.startswith("http"))

        data["es_portero"].append(int(gk))
        data["es_defensa"].append(int(de and not gk))
        data["es_centrocampista"].append(int(mid and not gk))
        data["es_delantero"].append(int(fw and not gk))
        data["juega_en_premier_league"].append(prem)
        data["juega_en_laliga"].append(lal)
        data["juega_en_serie_a"].append(ser)
        data["juega_en_bundesliga"].append(bun)
        data["juega_en_ligue1"].append(lig1)
        data["es_zurdo"].append(zurdo)
        data["es_diestro"].append(diestro)
        data["tiene_menos_de_25"].append(int(af is not None and af < 25))
        data["entre_25_y_30"].append(int(af is not None and 25 <= af <= 30))
        data["tiene_mas_de_30"].append(int(af is not None and af > 30))
        data["es_goleador"].append(int(g >= 8))
        data["tiene_muchas_asistencias"].append(int(a >= 4))
        data["es_joven_promesa"].append(int(af is not None and af < 23 and (fw or mid)))
        data["nacionalidad_europea"].append(int(b == "eu"))
        data["nacionalidad_sudamericana"].append(int(b == "sa"))
        data["nacionalidad_centroamerica_caribe"].append(int(b == "cc"))
        data["nacionalidad_norteamerica"].append(int(b == "na"))
        data["nacionalidad_africana"].append(int(b == "af"))
        data["nacionalidad_asia_u_oceania"].append(int(b == "ao"))
        data["altura_mayor_185"].append(int(hf is not None and hf >= 185))
        data["peso_mayor_82"].append(int(wf is not None and wf >= 82))
        data["ha_ganado_champions"].append(int(ucl_i >= 1))
        data["nacido_en_pais_big5"].append(int(nn in BIG5_NAT))
        data["juega_en_equipo_top_uefa"].append(_top_team(str(row.get("team_name") or "")))
        data["dorsal_menor_que_12"].append(int(sh_i is not None and sh_i < 12))
        data["dorsal_mayor_que_25"].append(int(sh_i is not None and sh_i > 25))
        data["rol_ofensivo_principal"].append(int(off_r))
        data["rol_defensivo_principal"].append(int(def_r))
        data["liga_norte_centro_europa"].append(liga_norte)
        data["liga_sur_europa"].append(liga_sur)
        data["veterano_33_o_mas"].append(int(af is not None and af >= 33))
        data["marcador_10_goles_o_mas"].append(int(g >= 10))
        data["asistente_5_o_mas"].append(int(a >= 5))
        data["pie_marcado_como_ambos"].append(ambos)
        data["tiene_imagen_api"].append(tiene_img)
        data["tiene_dorsal_registrado"].append(int(sh_i is not None))

        names.append(row["player_name"])
        photos.append(photo)
        pids.append(row["player_id_str"])

    X = pd.DataFrame(data)[BINARY_FEATURE_NAMES].fillna(0).astype(int)
    for c in X.columns:
        X[c] = X[c].clip(0, 1).astype(int)

    meta = pd.DataFrame({"player_id": pids, "player_name": names, "photo_url": photos})
    y = meta["player_id"].astype(str)
    if y.nunique() != len(y):
        y = meta["player_id"].astype(str) + "_" + meta["player_name"].astype(str)

    return X, y, meta
