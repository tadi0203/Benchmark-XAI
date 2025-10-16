# evaluation/Robustesse_SHAP.py
#  SHAP robustesse — modèle physique HDN/HDS
# Prenons le modèle physique comme reference appliqué sur données exprimentales simulées 
# ============================
#  BIBLIOTHÈQUES
# ============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
from tqdm import tqdm

from benchmark.physics import simulate_row
#  Chemins et config 
from benchmark.io import ROBUST, ensure_dir,DATA
from benchmark.config import  FEATURES,SEED   

# ----------------------------
#  Chemins E/S 
# ----------------------------
OLD_DATA_CSV = DATA / "simulated_results.csv"  
ROBUST_BASE_PATH      = ROBUST / "Robustesse_plots/shap"

# ----------------------------
#  Paramètres
# ----------------------------
RANDOM_STATE = SEED
np.random.seed(RANDOM_STATE)

PATH = OLD_DATA_CSV
SAVE_DIR = ROBUST_BASE_PATH

TARGETS = ["N", "S"]  # index 0 ou 1

BACKGROUND_SIZES = [10, 20, 40]
EXPLAIN_SIZES = [10, 20, 40]


# ----------------------------
#  Modèle physique 
# ----------------------------
class ModelePhysique:
    def predict(self, X: pd.DataFrame):
        preds = X.apply(simulate_row, axis=1)
        return np.vstack(preds.to_numpy())

model_physique = ModelePhysique()


# ----------------------------
#  SHAP avec background random
# ----------------------------
def save_shap_summary(model, X_train, X_test, feature_names, target, bg_size, exp_size, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    aux = 0 if target.upper() == "N" else 1

    # Background random
    n_bg = min(bg_size, len(X_train))
    X_bg = X_train.sample(n=n_bg, random_state=RANDOM_STATE)

    # Echantillon à expliquer
    n_exp = min(exp_size, len(X_test))
    X_exp = X_test.sample(n=n_exp, random_state=RANDOM_STATE)

    # Explainer
    f = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))[:, aux]
    explainer = shap.Explainer(f, X_bg, feature_names=feature_names, seed=RANDOM_STATE)

    # Valeurs SHAP
    sv = explainer(X_exp)

    # Summary plot
    plt.figure()
    shap.summary_plot(sv.values, X_exp, feature_names=feature_names, show=False)
    plt.tight_layout()

    out_dir = os.path.join(save_dir, target)
    # création du dossier 
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"SHAP_{target}_bg{n_bg}_exp{n_exp}.png"
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


# ----------------------------
#  Main
# ----------------------------
def main():
    df = pd.read_csv(PATH)
    X = df[FEATURES].copy()

    # Split 50/50
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=RANDOM_STATE, shuffle=True)

    results = []
    pbar_total = len(TARGETS) * len(BACKGROUND_SIZES) * len(EXPLAIN_SIZES)
    with tqdm(total=pbar_total, desc="SHAP Robustesse IFPEN") as pbar:
        for target in TARGETS:
            for bg_size in BACKGROUND_SIZES:
                for exp_size in EXPLAIN_SIZES:
                    try:
                        out = save_shap_summary(model_physique, X_train, X_test, FEATURES, target, bg_size, exp_size, SAVE_DIR)
                    except Exception as e:
                        results.append((target, bg_size, exp_size, None, f"ERROR: {e}"))
                    pbar.update(1)

    print("\n=== Résultats générés ===")
    for tgt, bg, exp, path, status in results:
        print(f"[{status}] target={tgt} | bg={bg} | exp={exp} -> {path}")

    print(f"\n Terminé. Graphes dans : {SAVE_DIR}")

if __name__ == "__main__":
    main()
