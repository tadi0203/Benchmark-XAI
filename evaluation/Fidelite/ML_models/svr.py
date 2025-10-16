# evaluation/Fidelite/ML_models/svr.py
# Entraînement SVR (pipeline StandardScaler + SVR) sur bdd perturbé

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import dump

# Chemins & config
from benchmark.io import DATA, ROBUST, ensure_dir
from benchmark.config import FEATURES, TARGETS, SEED  
# === Paramètres globaux 
TEST_SIZE = 0.2
CSV_PATH  = DATA / "perturbed_data.csv"   # bdd_ perturbé 
OUT_DIR   = ROBUST / "fidelite"/"ML_models"
PLOTS_DIR = OUT_DIR / "plots"
FEATURES=FEATURES+["aux_1","aux_2"]  # ajout des variables de perturbation (parasites)
# Grille de recherche 
PARAM_GRID = {
    "model__C": [1, 10, 50, 100],
    "model__epsilon": [0.1, 0.2, 0.3],
    "model__gamma": ["scale", "auto"],
    "model__kernel": ["rbf"],
}

def train_one_target(df: pd.DataFrame, target: str, save_dir: Path, seed: int,
                     test_size: float, make_parity_plot: bool = True):
    """
    Entraîne et évalue une pipeline (StandardScaler + SVR)
    Sauvegarde pipeline + parity plot. Retourne un dict de résultats.
    """
    ensure_dir(save_dir)

    X = df[FEATURES].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", SVR())
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    y_pred    = best_pipe.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))

    pipeline_path = save_dir / f"svm_{target}.joblib"
    dump(best_pipe, pipeline_path)

    parity_path = (PLOTS_DIR / f"parity_svm_{target}.png") if make_parity_plot else None
    if make_parity_plot:
        ensure_dir(PLOTS_DIR)
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        dmin = float(min(y_test.min(), y_pred.min()))
        dmax = float(max(y_test.max(), y_pred.max()))
        plt.plot([dmin, dmax], [dmin, dmax], "k--")
        plt.xlabel("Valeurs réelles")
        plt.ylabel("Valeurs prédites")
        plt.title(f"Parité - SVM Pipeline ({target})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(parity_path, dpi=200)
        plt.close()

    return {
        "target": target,
        "best_params": grid.best_params_,
        "r2": float(r2),
        "rmse": rmse,
        "mae": mae,
        "pipeline_path": str(pipeline_path),
        "parity_plot_path": str(parity_path) if parity_path else None,
    }

def main():
    np.random.seed(SEED)
    ensure_dir(OUT_DIR)
    ensure_dir(PLOTS_DIR)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Introuvable: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    results = []
    for target in TARGETS:  # ["N_sim", "S_sim"]
        print(f"\n=== Entraînement pour {target} ===")
        res = train_one_target(
            df=df,
            target=target,
            save_dir=OUT_DIR,
            seed=SEED,
            test_size=TEST_SIZE,
            make_parity_plot=True
        )
        results.append(res)
        print("  Meilleurs hyperparamètres :", res["best_params"])
        print(f"  R²   : {res['r2']:.4f}")
        print(f"  RMSE : {res['rmse']:.2f}")
        print(f"  MAE  : {res['mae']:.2f}")
        print("  Pipeline :", res["pipeline_path"])
        if res["parity_plot_path"]:
            print("  Parity plot :", res["parity_plot_path"])

    # Sauvegarde d’un récapitulatif CSV
    recap_path = OUT_DIR / "svm_pipelines_metrics.csv"
    pd.DataFrame([{
        "target": r["target"],
        "r2": r["r2"],
        "rmse": r["rmse"],
        "mae": r["mae"],
        "pipeline_path": r["pipeline_path"],
        "parity_plot_path": r["parity_plot_path"],
        "best_params": r["best_params"]
    } for r in results]).to_csv(recap_path, index=False)
    print("\nRécapitulatif métriques :", recap_path)

if __name__ == "__main__":
    main()
