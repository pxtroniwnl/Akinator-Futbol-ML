"""
Exporta una visualización del árbol entrenado a PDF (útil para informes / UTB).

Requisito: existir `artifacts/akinator_model.joblib` (entrena antes con la app o `model.train_and_save_from_csv`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from model import ARTIFACT_PATH, load_bundle
from preprocessing import BINARY_FEATURE_NAMES


def export_pdf(out: Path) -> None:
    bundle = load_bundle(ARTIFACT_PATH)
    fig, ax = plt.subplots(figsize=(28, 16))
    plot_tree(
        bundle.clf,
        feature_names=BINARY_FEATURE_NAMES,
        class_names=[str(i) for i in range(len(bundle.label_encoder.classes_))],
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=6,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("artifacts/arbol_akinator.pdf"))
    args = p.parse_args()
    export_pdf(args.output)
    print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
