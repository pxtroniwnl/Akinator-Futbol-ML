"""
Interfaz Streamlit del Akinator: recorre el árbol con Sí/No usando `st.session_state`.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from data_ingestion import ingest
from model import (
    ARTIFACT_PATH,
    RAW_CSV,
    get_next_question,
    load_bundle,
    resolve_player_from_leaf,
    step_on_answer,
    train_and_save_from_csv,
)

st.set_page_config(page_title="Akinator Fútbol", page_icon="⚽", layout="centered")


@st.cache_resource(show_spinner=False)
def ensure_bundle() -> None:
    """Garantiza CSV + modelo en disco (primera ejecución puede tardar por red)."""
    load_dotenv()
    if not RAW_CSV.exists():
        ingest(min_players=80, output_path=RAW_CSV)
    if not ARTIFACT_PATH.exists():
        train_and_save_from_csv(RAW_CSV, ARTIFACT_PATH)


def main() -> None:
    load_dotenv()
    st.title("Akinator de futbolistas")
    st.caption("Árbol de decisión entrenado con 40 características binarias (Scikit-Learn + Streamlit).")

    if st.button("Preparar / actualizar datos y modelo", help="Descarga CSV si falta y entrena el árbol."):
        with st.spinner("Trabajando…"):
            ingest(min_players=80, output_path=RAW_CSV)
            train_and_save_from_csv(RAW_CSV, ARTIFACT_PATH)
        st.cache_resource.clear()
        st.success("Listo. Recarga la página o pulsa **Nueva partida**.")
        return

    try:
        ensure_bundle()
    except Exception as e:
        st.error(f"No se pudo preparar el modelo: {e}")
        st.info("Configura `.env` (TheSportsDB suele funcionar con `THESPORTSDB_API_KEY=3`) o API-Football.")
        return

    bundle = load_bundle(ARTIFACT_PATH)

    if "node" not in st.session_state:
        st.session_state.node = 0
    if "history" not in st.session_state:
        st.session_state.history = []

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Nueva partida"):
            st.session_state.node = 0
            st.session_state.history = []
            st.rerun()

    info = get_next_question(bundle.clf, st.session_state.node)
    if info["terminal"]:
        name, url = resolve_player_from_leaf(bundle, st.session_state.node)
        st.success(f"¡Tu jugador es **{name}**!")
        if url and url.startswith("http"):
            try:
                st.image(url, width=280)
            except Exception:
                st.caption("No se pudo cargar la imagen desde la URL.")
        with st.expander("Camino de respuestas"):
            for h in st.session_state.history:
                st.write(h)
        return

    st.subheader("Responde")
    st.write(info["question"])
    with st.sidebar:
        st.metric("Reducción impureza (este nodo)", f"{info.get('impurity_decrease', 0):.2f}")
        st.caption("Valor pedagógico: cuánto baja la impureza ponderada en este split.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Sí", type="primary", use_container_width=True):
            st.session_state.history.append({"pregunta": info["question"], "respuesta": "Sí"})
            st.session_state.node = step_on_answer(bundle.clf, st.session_state.node, True)
            st.rerun()
    with c2:
        if st.button("No", use_container_width=True):
            st.session_state.history.append({"pregunta": info["question"], "respuesta": "No"})
            st.session_state.node = step_on_answer(bundle.clf, st.session_state.node, False)
            st.rerun()


if __name__ == "__main__":
    main()
