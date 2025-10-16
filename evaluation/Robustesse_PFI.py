# evaluation/Robustesse_PFI.py
#  PFI robustesse — modèle physique HDN/HDS
#  Prenons le modèle physique comme reference appliqué sur données exprimentales simulées 
# ============================
#  BIBLIOTHÈQUES
# ============================
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from benchmark.physics import simulate_row


#  Chemins et config 
from benchmark.io import ROBUST, ensure_dir,DATA
from benchmark.config import  FEATURES,SEED   

# ----------------------------
#  Chemins E/S 
# ----------------------------
OLD_DATA_CSV = DATA / "simulated_results.csv"  
ROBUST_BASE_PATH      = ROBUST / "Robustesse_plots/pfi/S"


# ----------------------------
# Paramètres
# ----------------------------
RANDOM_STATE = SEED
np.random.seed(RANDOM_STATE)
# Données & features 
PATH = OLD_DATA_CSV
# Où sauvegarder
SAVE_DIR = ROBUST_BASE_PATH

# Quelle sortie expliquer ? 0 = N, 1 = S
TARGET_INDEX = 1  # 0→N, 1→S


# Split
TEST_SIZE = 0.5  # 50/50 (≈ 92 lignes au total)

# Grilles de robustesse
N_REPEATS_LIST = [5, 10, 20, 50]
SCORINGS = ["neg_mse", "r2", "neg_mae"]  # métriques gérées ci-dessous

#TARGET_COL = "N_sim" ou "S_sim"
TARGET_COL = "S_sim"


# ----------------------------
# Utilitaires
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def predict_physical(X_df, which=0):
    """Retourne un vecteur scalaire (N ou S) à partir de simulate_row."""
    arr = np.vstack(X_df.apply(simulate_row, axis=1).to_numpy())  # (n, 2)
    return arr[:, which]

def score_vec(y_true, y_pred, scoring: str):
    """Renvoie un score scalaire (plus haut = mieux)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if scoring == "neg_mse":
        mse = np.mean((y_true - y_pred) ** 2)
        return -mse
    elif scoring == "neg_mae":
        mae = np.mean(np.abs(y_true - y_pred))
        return -mae
    elif scoring == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        raise ValueError(f"Scoring inconnu: {scoring}")

def permutation_importance_physical(X_test, y_true, predict_fn, n_repeats=20, scoring="neg_mse", random_state=42):
    """
    Calcule la PFI pour chaque feature :
    - baseline = score(predict_fn(X_test), y_true)
    - pour chaque feature j, répéter:
        * permuter aléatoirement X_test[j] (i.i.d. globale)
        * mesurer score permuté
      importance = baseline - score_permuté (moyenne et std sur n_repeats)
    Renvoie (imp_mean, imp_std) de taille n_features dans l'ordre de X_test.columns.
    """
    rng = np.random.RandomState(random_state)
    X_cols = list(X_test.columns)
    n_feat = len(X_cols)

    # Baseline
    y_base_pred = predict_fn(X_test)
    baseline = score_vec(y_true, y_base_pred, scoring)

    imp = np.zeros((n_repeats, n_feat), dtype=float)

    for r in range(n_repeats):
        for j, col in enumerate(X_cols):
            X_perm = X_test.copy()
            # permutation i.i.d. de la colonne j (perturbation "globale")
            X_perm[col] = rng.permutation(X_perm[col].values)
            y_perm_pred = predict_fn(X_perm)
            score_perm = score_vec(y_true, y_perm_pred, scoring)
            # importance = perte de score
            imp[r, j] = baseline - score_perm

    imp_mean = imp.mean(axis=0)
    imp_std = imp.std(axis=0, ddof=1) if n_repeats > 1 else np.zeros(n_feat)
    return imp_mean, imp_std

def plot_pfi(imp_df, title, out_png):
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["feature"], imp_df["pfi_mean"], xerr=imp_df["pfi_std"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance (baseline − score permuté)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(SAVE_DIR)

    # Données
    df = pd.read_csv(PATH)
    X = df[FEATURES].copy()

    # y_true : si colonne fournie, on l'utilise ; sinon on utilisera la prédiction baseline comme pseudo-vérité
    if TARGET_COL is not None and TARGET_COL in df.columns:
        y_all = df[TARGET_COL].values
    else:
        y_all = None

    # Split
    X_train, X_test = train_test_split(X, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    if y_all is not None:
        _, y_test = train_test_split(y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    else:
        y_test = None  # on le fixera à la baseline prédite

    # predict_fn pour N ou S
    predict_fn = lambda Xdf: predict_physical(Xdf, which=TARGET_INDEX)
    target_label = "N" if TARGET_INDEX == 0 else "S"

    # Si pas de y_test réel, on prend la baseline prédite (PFI reste valable pour tester la sensibilité structurelle)
    if y_test is None:
        y_test = predict_fn(X_test).copy()

    # Boucle robustesse
    for scoring in SCORINGS:
        for nrep in N_REPEATS_LIST:
            imp_mean, imp_std = permutation_importance_physical(
                X_test=X_test,
                y_true=y_test,
                predict_fn=predict_fn,
                n_repeats=nrep,
                scoring=scoring,
                random_state=RANDOM_STATE
            )

            imp_df = pd.DataFrame({
                "feature": X_test.columns,
                "pfi_mean": imp_mean,
                "pfi_std": imp_std
            }).sort_values("pfi_mean", ascending=False).reset_index(drop=True)

            tag = f"PFI_physical_{target_label}_{scoring}_rep{nrep}"
            out_dir = os.path.join(SAVE_DIR, target_label)
            # création du dossier 
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, f"{tag}.png")
            plot_pfi(imp_df, f"PFI — Physique ({target_label}) — scoring={scoring} — n_repeats={nrep}", out_png)
            
    print(f"\n Terminé. Graphes dans : {SAVE_DIR}")
if __name__ == "__main__":
    main()
