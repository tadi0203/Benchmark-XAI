# models/new_bdd/mlp_bdd_new.py
# Entraînement MLP sur la bdd optimale


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Chemins & config 
from benchmark.io import DATA, MODELS, ensure_dir
from benchmark.config import FEATURES, TARGETS, SEED

# Paramètres fixes
TEST_SIZE = 0.2
CSV_PATH  = DATA / "synthetic_dataset_filtered.csv"  # bdd optimale
OUT_DIR   = MODELS / "new_bdd"
PLOTS_DIR = OUT_DIR / "plots"

# Grille d'hyperparamètres 
PARAM_GRID = {
    "regressor__regressor__hidden_layer_sizes": [(64, 64), (128, 64), (128, 128)],
    "regressor__regressor__alpha": [1e-4, 1e-3, 1e-2],
    "regressor__regressor__learning_rate_init": [1e-3, 3e-3, 1e-2],
}

def train_and_evaluate_mlp_for_target(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    test_size: float = 0.2,
    seed: int = 42,
    make_parity_plot: bool = True,
):
    """
    Entraîne un MLPRegressor encapsulé dans TransformedTargetRegressor.
    Sauvegarde modèle + graphe de parité. Retourne métriques.
    """
    ensure_dir(OUT_DIR)
    ensure_dir(PLOTS_DIR)

    model_path  = OUT_DIR / f"mlp_{target}.joblib"
    parity_path = PLOTS_DIR / f"parity_mlp_{target}.png"

    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    x_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("regressor", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            early_stopping=True,
            n_iter_no_change=10,
            max_iter=500,
            tol=1e-4,
            random_state=seed,
            verbose=False,
        )),
    ])

    model = TransformedTargetRegressor(
        regressor=x_pipe,
        transformer=StandardScaler()
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    r2   = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))

    dump(best_model, model_path)

    if make_parity_plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        dmin = float(min(y_test.min(), y_pred.min()))
        dmax = float(max(y_test.max(), y_pred.max()))
        plt.plot([dmin, dmax], [dmin, dmax], "k--")
        plt.xlabel("Valeurs réelles")
        plt.ylabel("Valeurs prédites")
        plt.title(f"Parité - MLP ({target})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(parity_path, dpi=200)
        plt.close()
    else:
        parity_path = None

    return {
        "target": target,
        "best_params": grid.best_params_,
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
        print(f"\n=== Entraînement MLP pour {target} ===")
        res = train_and_evaluate_mlp_for_target(
            df=df,
            target=target,
            features=FEATURES,
            test_size=TEST_SIZE,
            seed=SEED,
            make_parity_plot=True,
        )
        results.append(res)
        print("  Meilleurs hyperparamètres :", res["best_params"])
        print(f"  R²   : {res['r2']:.4f}")
        print(f"  RMSE : {res['rmse']:.4f}")
        print(f"  MAE  : {res['mae']:.4f}")
        print("  Modèle :", res["model_path"])
        if res["parity_plot_path"]:
            print("  Parité :", res["parity_plot_path"])

    recap_path = OUT_DIR / "mlp_metrics_summary.csv"
    pd.DataFrame(results).to_csv(recap_path, index=False)
    print("\nRécapitulatif métriques :", recap_path)

if __name__ == "__main__":
    main()
