"""
Entrena un `DecisionTreeClassifier` con las 40 features binarias y expone navegación por `clf.tree_`.

La “siguiente mejor pregunta” en cada nodo interno es la **feature del split** que sklearn eligió
al ajustar el árbol (CART maximiza reducción de impureza local en cada paso).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from preprocessing import BINARY_FEATURE_NAMES, build_xy

ARTIFACT_PATH = Path("artifacts/akinator_model.joblib")
RAW_CSV = Path("data/raw_players.csv")

QUESTION_TEXT: dict[str, str] = {
    "es_portero": "¿Es portero?",
    "es_defensa": "¿Es defensa?",
    "es_centrocampista": "¿Es centrocampista?",
    "es_delantero": "¿Es delantero?",
    "juega_en_premier_league": "¿Juega en la Premier League?",
    "juega_en_laliga": "¿Juega en La Liga?",
    "juega_en_serie_a": "¿Juega en la Serie A?",
    "juega_en_bundesliga": "¿Juega en la Bundesliga?",
    "juega_en_ligue1": "¿Juega en la Ligue 1?",
    "es_zurdo": "¿Es zurdo?",
    "es_diestro": "¿Es diestro?",
    "tiene_menos_de_25": "¿Tiene menos de 25 años?",
    "entre_25_y_30": "¿Tiene entre 25 y 30 años?",
    "tiene_mas_de_30": "¿Tiene más de 30 años?",
    "es_goleador": "¿Es goleador (≥8 goles en los datos de temporada)?",
    "tiene_muchas_asistencias": "¿Tiene muchas asistencias (≥4 en los datos)?",
    "es_joven_promesa": "¿Es joven promesa (menos de 23 y medio/ataque)?",
    "nacionalidad_europea": "¿Tiene nacionalidad europea?",
    "nacionalidad_sudamericana": "¿Tiene nacionalidad sudamericana?",
    "nacionalidad_centroamerica_caribe": "¿Tiene nacionalidad centroamericana o caribeña?",
    "nacionalidad_norteamerica": "¿Tiene nacionalidad norteamericana?",
    "nacionalidad_africana": "¿Tiene nacionalidad africana?",
    "nacionalidad_asia_u_oceania": "¿Tiene nacionalidad asiática u oceánica?",
    "altura_mayor_185": "¿Mide 185 cm o más?",
    "peso_mayor_82": "¿Pesa 82 kg o más?",
    "ha_ganado_champions": "¿Figura como ganador de Champions en los trofeos cargados?",
    "nacido_en_pais_big5": "¿Su nacionalidad es Inglaterra, España, Italia, Francia o Alemania?",
    "juega_en_equipo_top_uefa": "¿Juega en un club de élite (heurística por nombre)?",
    "dorsal_menor_que_12": "¿Lleva dorsal menor que 12?",
    "dorsal_mayor_que_25": "¿Lleva dorsal mayor que 25?",
    "rol_ofensivo_principal": "¿Su rol es principalmente ofensivo?",
    "rol_defensivo_principal": "¿Su rol es principalmente defensivo?",
    "liga_norte_centro_europa": "¿Compite en liga inglesa o alemana?",
    "liga_sur_europa": "¿Compite en liga española o italiana?",
    "veterano_33_o_mas": "¿Tiene 33 años o más?",
    "marcador_10_goles_o_mas": "¿Marcó 10 goles o más en la temporada registrada?",
    "asistente_5_o_mas": "¿Dio 5 o más asistencias en la temporada registrada?",
    "pie_marcado_como_ambos": "¿Está marcado como ambidiestro?",
    "tiene_imagen_api": "¿Hay URL de imagen en la API?",
    "tiene_dorsal_registrado": "¿Tiene registrado un número de camiseta en los datos?",
}


def _check_questions() -> None:
    missing = [f for f in BINARY_FEATURE_NAMES if f not in QUESTION_TEXT]
    if missing:
        raise ValueError(f"Faltan textos de pregunta para: {missing}")


_check_questions()


def impurity_decrease_at_node(clf: DecisionTreeClassifier, node_id: int) -> float:
    tree = clf.tree_
    left, right = int(tree.children_left[node_id]), int(tree.children_right[node_id])
    if left == right:
        return 0.0
    n_p = float(tree.weighted_n_node_samples[node_id])
    imp_p = float(tree.impurity[node_id])
    n_l = float(tree.weighted_n_node_samples[left])
    imp_l = float(tree.impurity[left])
    n_r = float(tree.weighted_n_node_samples[right])
    imp_r = float(tree.impurity[right])
    return n_p * imp_p - n_l * imp_l - n_r * imp_r


def get_next_question(clf: DecisionTreeClassifier, node_id: int) -> dict[str, Any]:
    tree = clf.tree_
    left, right = int(tree.children_left[node_id]), int(tree.children_right[node_id])
    if left == right:
        return {"terminal": True, "node_id": node_id}

    fi = int(tree.feature[node_id])
    fname = BINARY_FEATURE_NAMES[fi]
    return {
        "terminal": False,
        "node_id": node_id,
        "feature_index": fi,
        "feature_name": fname,
        "question": QUESTION_TEXT[fname],
        "threshold": float(tree.threshold[node_id]),
        "impurity_decrease": impurity_decrease_at_node(clf, node_id),
    }


def step_on_answer(clf: DecisionTreeClassifier, node_id: int, answer_yes: bool) -> int:
    """
    Convención de la app: **Sí** = la característica es verdadera (valor 1 en el dataset).

    En sklearn, hijo izquierdo = muestras con `X[f] <= umbral`; con features 0/1 y umbral ~0.5,
    izquierda ≈ 0 (No), derecha ≈ 1 (Sí).
    """
    tree = clf.tree_
    left, right = int(tree.children_left[node_id]), int(tree.children_right[node_id])
    if left == right:
        return node_id
    return right if answer_yes else left


def leaf_class_index(clf: DecisionTreeClassifier, node_id: int) -> int:
    counts = clf.tree_.value[node_id][0]
    return int(np.argmax(counts))


@dataclass
class AkinatorModelBundle:
    clf: DecisionTreeClassifier
    label_encoder: LabelEncoder
    feature_names: list[str]
    meta: pd.DataFrame
    class_to_meta_row: dict[int, int]


def _class_to_first_row(y_enc: np.ndarray) -> dict[int, int]:
    m: dict[int, int] = {}
    for i, c in enumerate(y_enc):
        ci = int(c)
        if ci not in m:
            m[ci] = i
    return m


def train_bundle(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> AkinatorModelBundle:
    X = X[BINARY_FEATURE_NAMES]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(np.unique(y_enc))
    depth = min(14, max(4, int(np.ceil(np.log2(max(n_classes, 2))) + 4)))
    clf = DecisionTreeClassifier(
        random_state=42,
        max_depth=depth,
        min_samples_leaf=1,
        criterion="gini",
    )
    clf.fit(X.values, y_enc)
    return AkinatorModelBundle(
        clf=clf,
        label_encoder=le,
        feature_names=list(BINARY_FEATURE_NAMES),
        meta=meta.reset_index(drop=True),
        class_to_meta_row=_class_to_first_row(y_enc),
    )


def save_bundle(bundle: AkinatorModelBundle, path: Path = ARTIFACT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path = ARTIFACT_PATH) -> AkinatorModelBundle:
    return joblib.load(path)


def resolve_player_from_leaf(bundle: AkinatorModelBundle, node_id: int) -> tuple[str, str]:
    idx = leaf_class_index(bundle.clf, node_id)
    row_i = bundle.class_to_meta_row.get(idx, 0)
    m = bundle.meta.iloc[row_i]
    return str(m["player_name"]), str(m.get("photo_url") or "")


def train_and_save_from_csv(
    csv_path: Path = RAW_CSV,
    artifact_path: Path = ARTIFACT_PATH,
) -> AkinatorModelBundle:
    df = pd.read_csv(csv_path)
    X, y, meta = build_xy(df)
    bundle = train_bundle(X, y, meta)
    save_bundle(bundle, artifact_path)
    return bundle
