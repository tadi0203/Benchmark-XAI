# evaluation/Fidelite/ML_models/reg_poly.py
# Régression polynomiale sur la bdd perturbé

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Chemins & config 
from benchmark.io import DATA, ROBUST, ensure_dir
from benchmark.config import FEATURES, TARGETS, SEED

# Paramètres fixes
DEGREE    = 3
TEST_SIZE = 0.2

CSV_PATH  = DATA / "perturbed_data.csv"   # bdd_ perturbé 
OUT_DIR   = ROBUST / "fidelite"/"ML_models"
PLOTS_DIR = OUT_DIR / "plots"
FEATURES=FEATURES+["aux_1","aux_2"]  # ajout des variables de perturbation (parasites)

def train_and_evaluate_poly_for_target(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    degree: int = 2,
    test_size: float = 0.2,
    seed: int = 42,
    make_parity_plot: bool = True
):
    """
    Entraîne une pipeline (PolynomialFeatures -> StandardScaler -> LinearRegression)
    Sauvegarde le modèle et le graphe de parité. Retourne les métriques.
    """
    ensure_dir(OUT_DIR)
    ensure_dir(PLOTS_DIR)

    model_path  = OUT_DIR / f"polyreg_{target}_deg{degree}.joblib"
    parity_path = PLOTS_DIR / f"parity_polyreg_{target}_deg{degree}.png"

    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    pipe = Pipeline([
        ("poly",  PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("lin",   LinearRegression()),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2   = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))

    dump(pipe, model_path)

    if make_parity_plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        dmin = float(min(y_test.min(), y_pred.min()))
        dmax = float(max(y_test.max(), y_pred.max()))
        plt.plot([dmin, dmax], [dmin, dmax], "k--")
        plt.xlabel("Valeurs réelles")
        plt.ylabel("Valeurs prédites")
        plt.title(f"Parité - PolyReg (deg={degree}, target={target})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(parity_path, dpi=200)
        plt.close()
    else:
        parity_path = None

    return {
        "target": target,
        "degree": degree,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "model_path": str(model_path),
        "parity_plot_path": str(parity_path) if parity_path else None,
    }

def main():
    ensure_dir(OUT_DIR)
    ensure_dir(PLOTS_DIR)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Introuvable: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    results = []
    for target in TARGETS:  # ["N_sim", "S_sim"]
        print(f"\n=== Entraînement PolyReg (deg={DEGREE}) pour {target} ===")
        res = train_and_evaluate_poly_for_target(
            df=df,
            target=target,
            features=FEATURES,
            degree=DEGREE,
            test_size=TEST_SIZE,
            seed=SEED,
            make_parity_plot=True,
        )
        results.append(res)
        print(f"  R²   : {res['r2']:.4f}")
        print(f"  RMSE : {res['rmse']:.4f}")
        print(f"  MAE  : {res['mae']:.4f}")
        print("  Modèle :", res["model_path"])
        if res["parity_plot_path"]:
            print("  Parité :", res["parity_plot_path"])

    recap_path = OUT_DIR / f"polyreg_metrics_deg{DEGREE}.csv"
    pd.DataFrame(results).to_csv(recap_path, index=False)
    print("\nRécapitulatif métriques :", recap_path)

if __name__ == "__main__":
    main()
