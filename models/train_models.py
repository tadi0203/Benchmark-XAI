# models/train_models.py
# Entraîne SVM / MLP / Lasso / Poly / GBR sur l'ancienne BDD (simulated_results.csv)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from joblib import dump
from scipy.stats import uniform, randint, loguniform

#  Chemins et config 
from benchmark.io import DATA, MODELS, ensure_dir
from benchmark.config import FEATURES ,TARGETS 

# ----------------------------
#  Chemins E/S 
# ----------------------------
OLD_DATA_CSV = DATA / "simulated_results.csv"  
OUT_DIR      = MODELS / "old_bdd"
PLOTS_DIR    = OUT_DIR / "plots"

ensure_dir(OUT_DIR)
ensure_dir(PLOTS_DIR)

# ============================
#  Chargement des données simulées
# ============================
def load_data(path=OLD_DATA_CSV):
    """Charge les résultats simulés depuis un fichier CSV."""
    return pd.read_csv(path)

# ============================
#  Évaluation et visualisation d’un modèle
# ============================
def evaluate_model(yN_test, yS_test, yN_pred, yS_pred, save_dir=None, model_name="model", metrics_list=None):
    """
    Affiche les métriques et enregistre les graphiques de parité.

    - y*_test : valeurs réelles
    - y*_pred : prédictions du modèle
    - save_dir : dossier pour sauvegarde des figures
    - model_name : nom du modèle (ex. : "svm")
    - metrics_list : liste où stocker les métriques pour comparaison
    """
    # === Métriques de performance
    r2_N = r2_score(yN_test, yN_pred)
    r2_S = r2_score(yS_test, yS_pred)
    mae_N = mean_absolute_error(yN_test, yN_pred)
    mae_S = mean_absolute_error(yS_test, yS_pred)
    mape_N = np.mean(np.abs((yN_test - yN_pred) / yN_test)) * 100
    mape_S = np.mean(np.abs((yS_test - yS_pred) / yS_test)) * 100

    print("========== AZOTE (N) ==========")
    print("RMSE:", mean_squared_error(yN_test, yN_pred))
    print("R²:", r2_N)
    print("MAE:", mae_N)
    print("MedAE:", median_absolute_error(yN_test, yN_pred))
    print("MAPE:", mape_N)

    print("\n========== SOUFRE (S) ==========")
    print("RMSE:", mean_squared_error(yS_test, yS_pred))
    print("R²:", r2_S)
    print("MAE:", mae_S)
    print("MedAE:", median_absolute_error(yS_test, yS_pred))
    print("MAPE:", mape_S)

    if metrics_list is not None:
        metrics_list.append({
            "model": model_name,
            "R2_N": r2_N,
            "R2_S": r2_S,
            "MAE_N": mae_N,
            "MAE_S": mae_S,
            "MAPE_N": mape_N,
            "MAPE_S": mape_S
        })

    # === Graphique de parité : Azote
    plt.figure(figsize=(6, 6))
    plt.scatter(yN_test, yN_pred, alpha=0.7)
    plt.plot([min(yN_test), max(yN_test)], [min(yN_test), max(yN_test)], 'k--')
    plt.xlabel("Azote réel (ppm)")
    plt.ylabel("Azote prédit (ppm)")
    plt.title(f"Parité Azote – R² = {r2_N:.2f}")
    plt.grid()
    plt.tight_layout()
    if save_dir:
        path_N = os.path.join(save_dir, f"{model_name}_N_parity.png")
        plt.savefig(path_N)
        print(f"[✓] Graphe de parité N enregistré : {path_N}")
    plt.close()

    # === Graphique de parité : Soufre
    plt.figure(figsize=(6, 6))
    plt.scatter(yS_test, yS_pred, alpha=0.7)
    plt.plot([min(yS_test), max(yS_test)], [min(yS_test), max(yS_test)], 'k--')
    plt.xlabel("Soufre réel (ppm)")
    plt.ylabel("Soufre prédit (ppm)")
    plt.title(f"Parité Soufre – R² = {r2_S:.2f}")
    plt.grid()
    plt.tight_layout()
    if save_dir:
        path_S = os.path.join(save_dir, f"{model_name}_S_parity.png")
        plt.savefig(path_S)
        print(f"[✓] Graphe de parité S enregistré : {path_S}")
    plt.close()

# ============================
#  Résumé des métriques pour tous les modèles
# ============================
def summarize_metrics(metrics_list,
                      save_csv=OUT_DIR / "summary_metrics.csv",
                      save_plot=OUT_DIR / "plots" / "r2_barplot.png"):
    """
    Sauvegarde un résumé CSV des métriques et un barplot comparatif R².
    """
    df = pd.DataFrame(metrics_list)
    ensure_dir(Path(save_csv).parent)
    ensure_dir(Path(save_plot).parent)
    df.to_csv(save_csv, index=False)

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(df['model']))
    plt.bar(x - bar_width/2, df['R2_N'], width=bar_width, label='Azote (N)')
    plt.bar(x + bar_width/2, df['R2_S'], width=bar_width, label='Soufre (S)')
    plt.xticks(x, df['model'])
    plt.ylabel("R²")
    plt.title("Scores R² des modèles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_plot)
    plt.close()
    print(f"[✓] Barplot R² enregistré : {save_plot}\n[✓] Résumé CSV enregistré : {save_csv}")

# ============================
#  Entraînement avec choix du modèle
# ============================
def train_model(Xtrain, yN_train, yS_train, model_type="svm"):
    """
    Entraîne un modèle de régression pour N et S selon le type choisi.

    model_type : ["svm", "mlp", "lasso", "poly", "gbr"]
    """
    if model_type == "svm":
        pipeline_N = make_pipeline(MinMaxScaler(), SVR())
        pipeline_S = make_pipeline(MinMaxScaler(), SVR())
        param_N = {
            'svr__C': uniform(0.1, 1e6),
            'svr__epsilon': uniform(0.01, 100),
            'svr__kernel': ['linear', 'rbf', 'poly'],
            'svr__gamma': ['scale', 'auto']
        }
        search_N = RandomizedSearchCV(pipeline_N, param_N, n_iter=20, scoring='neg_mean_squared_error', cv=5, verbose=1, random_state=42)
        search_S = RandomizedSearchCV(pipeline_S, param_N, n_iter=20, scoring='neg_mean_squared_error', cv=5, verbose=1, random_state=42)

    elif model_type == "mlp":
        pipeline_N = make_pipeline(MinMaxScaler(), MLPRegressor(max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=42))
        pipeline_S = make_pipeline(MinMaxScaler(), MLPRegressor(max_iter=5000, early_stopping=True, n_iter_no_change=20, random_state=42))
        param_N = {
            "mlpregressor__hidden_layer_sizes": [(8,), (12,), (16,), (24,), (8, 4), (16, 8)],
            "mlpregressor__activation": ["relu", "tanh", "logistic"],
            "mlpregressor__solver": ["adam", "lbfgs"],
            "mlpregressor__alpha": loguniform(1e-6, 1e-2),
            "mlpregressor__learning_rate_init": loguniform(5e-5, 5e-3),
            "mlpregressor__learning_rate": ["constant", "adaptive"],
            "mlpregressor__batch_size": ["auto", 8, 16, 32]
        }
        param_S = {
            "mlpregressor__hidden_layer_sizes": [(16,32),(8,16),(32,128),(64,128),(64,32),(16,32),(32,16)],
            "mlpregressor__activation": ["relu", "tanh", "logistic"],
            "mlpregressor__solver": ["adam", "lbfgs"],
            "mlpregressor__alpha": loguniform(1e-6, 1e-2),
            "mlpregressor__learning_rate_init": loguniform(5e-5, 5e-2),
            "mlpregressor__learning_rate": ["constant", "adaptive"],
            "mlpregressor__batch_size": ["auto", 8, 16, 32]
        }
        search_N = RandomizedSearchCV(pipeline_N, param_N, n_iter=50, scoring='neg_mean_squared_error', cv=5, random_state=42, verbose=1)
        search_S = RandomizedSearchCV(pipeline_S, param_S, n_iter=50, scoring='neg_mean_squared_error', cv=5, random_state=42, verbose=1)

    elif model_type == "lasso":
        pipeline_N = make_pipeline(MinMaxScaler(), Lasso())
        pipeline_S = make_pipeline(MinMaxScaler(), Lasso())
        param = {"lasso__alpha": loguniform(1e-5, 1e3)}
        search_N = RandomizedSearchCV(pipeline_N, param, n_iter=20, scoring="neg_mean_absolute_error", cv=5, verbose=1)
        search_S = RandomizedSearchCV(pipeline_S, param, n_iter=20, scoring="neg_mean_absolute_error", cv=5, verbose=1)

    elif model_type == "poly":
        pipeline_N = make_pipeline(MinMaxScaler(), PolynomialFeatures(), Ridge())
        pipeline_S = make_pipeline(MinMaxScaler(), PolynomialFeatures(), Lasso(max_iter=10000))
        param_N = {
            "polynomialfeatures__degree": randint(1, 12),
            "ridge__alpha": loguniform(1e-4, 1e2)
        }
        param_S = {
            "polynomialfeatures__degree": randint(1, 3),
            "lasso__alpha": loguniform(1e-1, 1)
        }
        search_N = RandomizedSearchCV(pipeline_N, param_N, cv=5, scoring='neg_mean_squared_error', verbose=1, random_state=42)
        search_S = RandomizedSearchCV(pipeline_S, param_S, cv=5, scoring='neg_mean_squared_error', verbose=1, random_state=42)

    elif model_type == "gbr":
        pipeline = make_pipeline(MinMaxScaler(), GradientBoostingRegressor(random_state=42))
        param = {
            "gradientboostingregressor__n_estimators": [50, 80, 100, 200, 300, 500],
            "gradientboostingregressor__max_depth": [3, 5, 7, 9],
            "gradientboostingregressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "gradientboostingregressor__subsample": [0.6, 0.8, 1.0],
            "gradientboostingregressor__min_samples_split": [2, 5, 10],
            "gradientboostingregressor__min_samples_leaf": [1, 2, 5, 10]
        }
        search_N = GridSearchCV(pipeline, param, scoring='r2', cv=5, n_jobs=-1, verbose=2)
        search_S = GridSearchCV(pipeline, param, scoring='r2', cv=5, n_jobs=-1, verbose=2)

    else:
        raise ValueError(f"Modèle inconnu : {model_type}")

    # Entraînement
    model_N = search_N.fit(Xtrain, yN_train).best_estimator_
    model_S = search_S.fit(Xtrain, yS_train).best_estimator_
    return model_N, model_S

# ============================
#  Pipeline principal
# ============================
if __name__ == "__main__":
    df = load_data(OLD_DATA_CSV)

    # Features : on garde Nom_charge pour la stratification, puis on la supprime de X
    features = ["Nom_charge"] + FEATURES  # FEATURES vient de benchmark.config (sans "Nom_charge")
    X = df[features].copy()
    y_N = df["N_sim"].copy()
    y_S = df["S_sim"].copy()
    groups = df["Nom_charge"].copy()  # Pour stratification

    # Split avec stratification par charge
    Xtrain, Xtest, yN_train, yN_test = train_test_split(X, y_N, test_size=0.2, stratify=groups, random_state=42)
    _,      _,      yS_train, yS_test = train_test_split(X, y_S, test_size=0.2, stratify=groups, random_state=42)

    # Supprimer la colonne Nom_charge (non numérique) avant l'entraînement
    Xtrain = Xtrain.drop(columns=["Nom_charge"]).copy()
    Xtest  = Xtest.drop(columns=["Nom_charge"]).copy()

    # Liste des métriques
    metrics_list = []

    # Entraînement de plusieurs modèles (inchangés)
    for model in ["svm", "mlp", "lasso", "poly", "gbr"]:
        print(f"\n===== Entraînement du modèle : {model.upper()} =====")
        model_N, model_S = train_model(Xtrain, yN_train, yS_train, model_type=model)
        yN_pred = model_N.predict(Xtest)
        yS_pred = model_S.predict(Xtest)

        # Évalue + enregistre les plots dans models/old_bdd/plots
        evaluate_model(yN_test, yS_test, yN_pred, yS_pred,
                       save_dir=str(PLOTS_DIR), model_name=model, metrics_list=metrics_list)

        # Sauvegarde des modèles entraînés dans models/old_bdd/
        dump(model_N, OUT_DIR / f"modele_{model}_N_sim.joblib")
        dump(model_S, OUT_DIR / f"modele_{model}_S_sim.joblib")

    # Résumé des métriques (CSV + barplot)
    summarize_metrics(metrics_list,
                      save_csv=OUT_DIR / "summary_metrics.csv",
                      save_plot=PLOTS_DIR / "r2_barplot.png")
