"""
Interfaz Streamlit del Akinator: recorre el árbol con Sí/No usando `st.session_state`.

Evita importar pandas/sklearn al arrancar el proceso: así la primera pintura no queda
minutos en el esqueleto de carga (especialmente en WSL + disco /mnt/c).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Akinator Fútbol", page_icon="⚽", layout="centered")


def _project_root() -> Path:
    return Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Cargando modelo en memoria…")
def _bundle_cached(artifact_path_str: str):
    from model import load_bundle

    return load_bundle(Path(artifact_path_str))


def main() -> None:
    load_dotenv()
    root = _project_root()
    raw_csv = root / "data" / "raw_players.csv"
    artifact = root / "artifacts" / "akinator_model.joblib"

    st.title("Akinator de futbolistas")
    st.caption("Árbol de decisión entrenado con 40 características binarias (Scikit-Learn + Streamlit).")

    csv_ok = raw_csv.exists()
    model_ok = artifact.exists()

    if not (csv_ok and model_ok):
        st.warning(
            "Faltan archivos para jugar "
            f"(CSV={csv_ok}, modelo={model_ok}). "
            f"Buscados en:\n- `{raw_csv}`\n- `{artifact}`"
        )
        st.info(
            "Pulsa el botón para descargar datos y entrenar. **Puede tardar varios minutos** "
            "(muchas llamadas a la API)."
        )
        if st.button("Preparar datos y entrenar modelo"):
            try:
                with st.spinner("Descargando datos y entrenando… (no cierres esta pestaña)"):
                    from data_ingestion import ingest
                    from model import train_and_save_from_csv

                    load_dotenv()
                    if not raw_csv.exists():
                        ingest(min_players=80, output_path=raw_csv)
                    if not artifact.exists():
                        train_and_save_from_csv(raw_csv, artifact)
                    st.cache_resource.clear()
                st.success("Listo. Recargando…")
                st.rerun()
            except Exception as e:
                st.error(f"No se pudo preparar el modelo: {e}")
                st.info("Revisa red, firewall y `.env` (TheSportsDB suele ir con `THESPORTSDB_API_KEY=3`).")
        st.stop()

    bundle = _bundle_cached(str(artifact))

    if "node" not in st.session_state:
        st.session_state.node = 0
    if "history" not in st.session_state:
        st.session_state.history = []

    from model import get_next_question, resolve_player_from_leaf, step_on_answer

    col_a, _ = st.columns(2)
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
