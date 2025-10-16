# interpretability/Explainability_Analysis.py
# Explicabilité : SHAP, PFI, PDP, LIME  sur  SVR, MLP, Lasso, poly, gbr entrainée entrainée sur old data set
# ============================
#  BIBLIOTHÈQUES
# ============================
import os
import joblib
from joblib import load
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import train_test_split

#  Chemins et config 
from benchmark.io import MODELS, XAI, ensure_dir,DATA
from benchmark.config import  TARGETS,FEATURES,SEED   

# ----------------------------
#  Chemins E/S 
# ----------------------------
MODEL_BASE_PATH = MODELS / "old_bdd"  
OLD_DATA_CSV = DATA / "simulated_results.csv"  
EXPLAIN_BASE_PATH      = XAI / "Analysis"

ensure_dir(EXPLAIN_BASE_PATH)

features = ["Nom_charge"] + FEATURES 
# ============================
#  SHAP Summary Plot
# ============================
def save_shap_summary(model, Xtrain,Xtest, name, save_dir):
    """
    Calcule et sauvegarde un graphe SHAP global (summary plot).
    """
    explainer = shap.Explainer(model.predict, Xtrain)
    shap_values = explainer(Xtest)
    plt.figure()
    # beeswarm plot 
    shap.summary_plot(shap_values, Xtest, show=False)
    path = os.path.join(save_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ============================
#  Permutation Feature Importance
# ============================
def save_pfi(model, X, y, name, save_dir):
    """
    Calcule l’importance des variables par permutation (PFI) avec std et sauvegarde le barplot.
    """
    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        n_repeats=20,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1   
    )
    importances = result.importances_mean
    errors = result.importances_std

    imp_df = pd.DataFrame({
        'feature': X.columns,
        'pfi_mean': importances,
        'pfi_std': errors
    }).sort_values('pfi_mean', ascending=False)
    # importance globale plot pfi
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df['feature'], imp_df['pfi_mean'], xerr=imp_df['pfi_std'])
    plt.gca().invert_yaxis()
    plt.xlabel("Baisse moyenne de la performance")
    plt.title("Permutation Feature Importance")
    path = os.path.join(save_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ============================
#  Partial Dependence Plots (PDP)
# ============================
def save_pdp(model, X, features, name, save_dir):
    """
    Sauvegarde les partial dependence plots (PDP) pour les variables données.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(model, X, features=features, ax=ax)
    path = os.path.join(save_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ============================
#  LIME – Importance Globale + Beeswarm
# ============================
def save_lime(model, Xtrain, Xtest, name, save_dir):
    """
    Applique LIME sur toutes les instances de Xtest et sauvegarde :
    - l’importance globale moyenne
    - un graphe "beeswarm" colorié selon la valeur de la variable
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=Xtrain.values,
        feature_names=Xtrain.columns.tolist(),
        mode="regression",
        random_state=42
    )

    def _return_weights(exp_lime):
        exp_list = sorted(exp_lime.as_map()[1], key=lambda t: t[0])
        return np.array([w for _, w in exp_list])

    n_instances = len(Xtest)
    weights = []
    for i in range(n_instances):
        explanation = explainer.explain_instance(
            data_row=Xtest.iloc[i].values,
            predict_fn=model.predict,
            num_features=Xtest.shape[1]
        )
        weights.append(_return_weights(explanation))

    lime_weights = pd.DataFrame(weights, columns=Xtest.columns)
    abs_mean_df = lime_weights.abs().mean().sort_values().reset_index()
    abs_mean_df.columns = ["feature", "abs_mean"]

    #  Barplot global des poids absolus moyens
    fig, ax = plt.subplots(figsize=(8, 4))
    yticks = range(len(abs_mean_df))
    ax.barh(yticks, abs_mean_df["abs_mean"])
    ax.set_yticks(yticks)
    ax.set_yticklabels(abs_mean_df["feature"], fontsize=12)
    ax.set_xlabel("Mean |LIME weight|", fontsize=14)
    ax.set_title("Importance globale (LIME)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{name}_global.png"))
    plt.close(fig)

    #  Beeswarm plot LIME
    fig, ax = plt.subplots(figsize=(8, 4))
    ylabels = abs_mean_df["feature"]
    for i, feat in enumerate(ylabels):
        x_vals = lime_weights[feat]
        c_vals = Xtest[feat][:n_instances]
        ax.scatter(x_vals, [i]*len(x_vals), c=c_vals, cmap="bwr", edgecolors="k", alpha=0.8)

    ax.vlines(0, ymin=-1, ymax=len(ylabels), colors="k", linestyles="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_xlabel("LIME weight", fontsize=14)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Feature value", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{name}_beeswarm.png"))
    plt.close(fig)

# ============================
#  Appliquer toutes les méthodes XAI à un modèle
# ============================
def explain_model(path_model, Xtrain, Xtest, y, name, features):
    """
    Charge un modèle et applique toutes les méthodes d’explication :
    SHAP, PFI, PDP, LIME.
    """
    model = joblib.load(path_model)
    save_dir = os.path.join(EXPLAIN_BASE_PATH, name)
    os.makedirs(save_dir, exist_ok=True)

    save_shap_summary(model, Xtrain,Xtest, f"{name}_shap", save_dir)
    save_pfi(model, Xtest, y, f"{name}_pfi", save_dir)
    save_pdp(model, Xtest, features, f"{name}_pdp", save_dir)
    save_lime(model, Xtrain, Xtest, f"{name}_lime", save_dir)

# ============================
#  Pipeline d’explication pour tous les modèles (N/S)
# ============================
def explain_all_models(Xtrain, Xtest, yN_test, yS_test):
    """
    Applique explain_model() pour tous les modèles entraînés (N et S).
    """
    features = Xtest.columns.tolist()
    for target, y_test in zip(TARGETS, [yN_test, yS_test]):
        for model_name in ["svm", "lasso", "poly","gbr","mlp"]:
            path = os.path.join(MODEL_BASE_PATH, f"modele_{model_name}_{target}.joblib")
            if os.path.exists(path):
                print(f"[→] Explicabilité : {model_name.upper()} pour {target}")
                explain_model(path, Xtrain, Xtest, y_test, f"{model_name}_{target}", features)
            else:
                print(f"[!] Modèle non trouvé : {path}")






# ============================
#  Chargement des données simulées
# ============================
df = pd.read_csv(OLD_DATA_CSV)

# Sélection des features utilisées lors de l'entraînement
X = df[features]
y_N = df["N_sim"]   # Cible Azote
y_S = df["S_sim"]   # Cible Soufre
groups = df["Nom_charge"]  # Utilisé pour la stratification

# ============================
#  Séparation entraînement/test avec stratification par charge
# ============================
Xtrain, Xtest, yN_train, yN_test = train_test_split(X, y_N, test_size=0.2, stratify=groups, random_state=SEED)
_, _, yS_train, yS_test       = train_test_split(X, y_S, test_size=0.2, stratify=groups, random_state=SEED)

# Retirer la variable "Nom_charge" car elle est catégorielle
Xtrain = Xtrain.drop(columns=["Nom_charge"])
Xtest  = Xtest.drop(columns=["Nom_charge"])

# ============================
#  Lancement de l’explication pour tous les modèles
# ============================
# Cette fonction va automatiquement :
# - Charger les modèles `svm`, `RN`, `lasso`, `poly` (N et S)
# - Appliquer SHAP, LIME, PFI, PDP
# - Sauvegarder les résultats dans `EXPLAIN_BASE_PATH`
explain_all_models(Xtrain, Xtest, yN_test, yS_test)