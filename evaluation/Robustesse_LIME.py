# evaluation/Robustesse_LIME.py
#  LIME robustesse — modèle physique HDN/HDS
# Prenons le modèle physique comme reference appliqué sur données exprimentales simulées 
# ============================
#  BIBLIOTHÈQUES
# ============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lime.lime_tabular as llt

from benchmark.physics import simulate_row
#  Chemins et config 
from benchmark.io import ROBUST, ensure_dir,DATA
from benchmark.config import  FEATURES,SEED   

# ----------------------------
#  Chemins E/S 
# ----------------------------
OLD_DATA_CSV = DATA / "simulated_results.csv"  
ROBUST_BASE_PATH      = ROBUST / "Robustesse_plots/lime"

# ----------------------------
# Paramètres
# ----------------------------
RANDOM_STATE = SEED
np.random.seed(RANDOM_STATE)

# Données 
PATH = OLD_DATA_CSV
# sortie
SAVE_DIR = ROBUST_BASE_PATH
# Split
TEST_SIZE = 0.5
# On explique la sortie N (= index 0) ou S (= index 1)
TARGET_INDEX = 0   
# Nombre d'instances expliquées  et n_samples LIME
N_INSTANCES_MAX = 30
NUM_SAMPLES_LIST = [500, 1000, 1500]

# Schémas de perturbation 
#  - global : comportement LIME par défaut (perturbations globales, quartiles)
#  - local  : perturbations centrées sur l'instance 
PERTURBATION_SCHEMES = [
    ("global", False),  # sample_around_instance=False
    ("local",  True),   # sample_around_instance=True
]


# ----------------------------
# Fonctions utilitaires
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def predict_fn_physical(which=0):
    """Wrapper scalaire: renvoie la sortie N (0) ou S (1) du modèle physique."""
    def _f(X_np):
        X_df = pd.DataFrame(X_np, columns=FEATURES)
        arr = np.vstack(X_df.apply(simulate_row, axis=1).to_numpy())  
        return arr[:, which]
    return _f

def compute_lime_weights_matrix(explainer, Xtest, predict_fn, num_features, num_samples, n_instances_limit):
    """Retourne une matrice (n_instances x n_features) des poids LIME."""
    n_instances = min(n_instances_limit, len(Xtest))
    weights = []
    feat_count = Xtest.shape[1]

    for i in range(n_instances):
        exp = explainer.explain_instance(
            data_row     = Xtest.iloc[i].values,
            predict_fn   = predict_fn,
            num_features = num_features,
            num_samples  = num_samples
        )
        # Remise en ordre des poids par index de feature
        full = np.zeros(feat_count)
        for idx, w in sorted(exp.as_map()[1], key=lambda t: t[0]):
            if 0 <= idx < feat_count:
                full[idx] = w
        weights.append(full)

    return np.vstack(weights)  # (n_instances, n_features)

def save_importance_bar(abs_mean_series, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = abs_mean_series.sort_values()
    ax.barh(range(len(ordered)), ordered.values)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered.index, fontsize=10)
    ax.set_xlabel("Mean |LIME weight|", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_beeswarm(weights_df, Xref, out_path, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    feats = list(weights_df.columns)
    for i, feat in enumerate(feats):
        x_vals = weights_df[feat].values
        c_vals = Xref[feat].head(len(x_vals)).values
        ax.scatter(x_vals, np.full_like(x_vals, i, dtype=float),
                   c=c_vals, cmap="bwr", edgecolors="k", alpha=0.8)
    ax.vlines(0, ymin=-1, ymax=len(feats), colors="k", linestyles="--")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=10)
    ax.set_xlabel("LIME weight", fontsize=11)
    ax.set_title(title, fontsize=12)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Feature value", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(SAVE_DIR)

    # Données
    df = pd.read_csv(PATH)
    X  = df[FEATURES].copy()

    # Split 50/50
    Xtrain, Xtest = train_test_split(X, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

    # Modèle physique 
    predict_fn = predict_fn_physical(which=TARGET_INDEX)
    model_label = f"physical_target_{'N' if TARGET_INDEX==0 else 'S'}"

    # Boucles   : schéma de perturbation x num_samples
    for scheme_name, sample_local in PERTURBATION_SCHEMES:
        # Explainer LIME minimal
        explainer = llt.LimeTabularExplainer(
            training_data          = Xtrain.values,
            feature_names          = FEATURES,
            mode                   = "regression",
            random_state           = RANDOM_STATE,
            discretize_continuous  = True,       # défaut (quartiles)
            discretizer            = "quartile", # explicite
            sample_around_instance = sample_local
        )

        for num_samples in NUM_SAMPLES_LIST:
            # Matrice des poids LIME
            W = compute_lime_weights_matrix(
                explainer         = explainer,
                Xtest             = Xtest,
                predict_fn        = predict_fn,
                num_features      = Xtest.shape[1],
                num_samples       = num_samples,
                n_instances_limit = N_INSTANCES_MAX
            )

            weights_df = pd.DataFrame(W, columns=Xtest.columns)
            abs_mean   = weights_df.abs().mean()

            tag   = f"{model_label}__{scheme_name}__ns{num_samples}"
            title = f"LIME — {model_label} — {scheme_name} — n_samples={num_samples}"
            target=f"{'N' if TARGET_INDEX==0 else 'S'}"
            out_dir = os.path.join(SAVE_DIR, target)
            # création du dossier 
            os.makedirs(out_dir, exist_ok=True)
            out_bar   = os.path.join(out_dir, f"LIME_{tag}_bar.png")
            out_swarm = os.path.join(out_dir, f"LIME_{tag}_swarm.png")

            save_importance_bar(abs_mean, out_bar, title + " (global importance)")
            save_beeswarm(weights_df, Xtest, out_swarm, title + " (beeswarm)")



    print(f"\n Terminé. Graphes dans : {SAVE_DIR}")


if __name__ == "__main__":
    main()
