# interpretability/new_bdd_XAI/MLP_BDD_NEW_EXP.py
# Explicabilité : SHAP, PFI, PDP, LIME sur MLP entrainée sur la nouvelle bdd
# ============================
#  BIBLIOTHÈQUES
# ============================
import os
import joblib
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

#  Chemins et config 
from benchmark.io import MODELS, XAI, ensure_dir,DATA
from benchmark.config import  FEATURES,SEED   

# ----------------------------
#  Chemins E/S 
# ----------------------------
MODEL_BASE_PATH = MODELS / "new_bdd"  
NEW_DATA_CSV = DATA / "synthetic_dataset_filtered.csv"  
EXPLAIN_BASE_PATH      = XAI / "Analysis_new_bdd\MLP"
ensure_dir(EXPLAIN_BASE_PATH)


# ===========================
#  Chemins & paramètres
# ============================
# Données
CSV_PATH = NEW_DATA_CSV 
TEST_SIZE = 0.2
# Sous-échantillons pour XAI
BG_SIZE = 100       # nb. de points train pour “entraîner” les explaineurs
TEST_SUB_SIZE = 10  # nb. de points test à expliquer

# ============================
# utilitaires
# ============================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def model_path_for_target(target: str) -> str:
    # Nom des fichiers modèles MLP
    return os.path.join(MODEL_BASE_PATH, f"mlp_{target}.joblib")
def target_out_dirs(target: str):
    base = os.path.join(EXPLAIN_BASE_PATH, target)
    shap_dir = os.path.join(base, "SHAP")
    pfi_dir  = os.path.join(base, "PFI")
    pdp_dir  = os.path.join(base, "PDP")
    lime_dir = os.path.join(base, "LIME")
    for d in (base, shap_dir, pfi_dir, pdp_dir, lime_dir):
        ensure_dir(d)
    return shap_dir, pfi_dir, pdp_dir, lime_dir

def sample_train_test_subsets(X_train, X_test, y_train, y_test,
                              bg_size=BG_SIZE, test_sub_size=TEST_SUB_SIZE, seed=SEED):
    """Échantillonne 100 points du train (background) et 10 points du test (à expliquer)."""
    bg_n = min(bg_size, len(X_train))
    ts_n = min(test_sub_size, len(X_test))
    X_bg   = X_train.sample(n=bg_n, random_state=seed)
    y_bg   = y_train.loc[X_bg.index]
    X_sub  = X_test.sample(n=ts_n, random_state=seed)
    y_sub  = y_test.loc[X_sub.index]
    return X_bg, y_bg, X_sub, y_sub

# ============================
#  SHAP Summary Plot (bg=100, test=10) via KernelExplainer
# ============================
def save_shap_summary(model, X_bg: pd.DataFrame, X_test_sub: pd.DataFrame, name: str, save_dir: str):
    """
    SHAP pour MLP : KernelExplainer 
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_names = list(X_bg.columns)

    def predict_np(X_np: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X_np, columns=feature_names)
        yhat = model.predict(X_df)
        return np.asarray(yhat).reshape(-1)

    explainer = shap.KernelExplainer(predict_np, X_bg.values, link="identity")
    shap_values = explainer.shap_values(X_test_sub.values, nsamples="auto")

    plt.figure()
    shap.summary_plot(shap_values, X_test_sub.values, feature_names=feature_names, show=False)
    out = os.path.join(save_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out

# ============================
#  Permutation Feature Importance (sur 10 points test)
# ============================
def save_pfi(model, X_test_sub: pd.DataFrame, y_test_sub: pd.Series, name: str, save_dir: str):
    result = permutation_importance(
        estimator=model,
        X=X_test_sub,
        y=y_test_sub,
        n_repeats=20,
        scoring="neg_mean_squared_error",
        random_state=SEED,
        n_jobs=-1
    )
    imp_df = pd.DataFrame({
        "feature": X_test_sub.columns,
        "pfi_mean": result.importances_mean,
        "pfi_std":  result.importances_std
    }).sort_values("pfi_mean", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["feature"], imp_df["pfi_mean"], xerr=imp_df["pfi_std"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean decrease in performance (perm.)")
    plt.title("Permutation Feature Importance (MLP) — test subset (n=10)")
    out = os.path.join(save_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out

# ============================
#  Partial Dependence Plots (PDP) 1D (sur 100 points train)
# ============================
def save_pdp(model, X_bg: pd.DataFrame, features: list[str], name: str, save_dir: str):
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(model, X_bg, features=features, ax=ax, kind="average")
    out = os.path.join(save_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    return out

# ============================
#  LIME – Importance Globale + Beeswarm (bg=100, test=10)
# ============================
def save_lime(model, X_bg: pd.DataFrame, X_test_sub: pd.DataFrame, name: str, save_dir: str):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_bg.values,
        feature_names=X_bg.columns.tolist(),
        mode="regression",
        random_state=SEED,
        discretize_continuous=True
    )

    def _return_weights(exp_lime):
        exp_list = sorted(exp_lime.as_map()[1], key=lambda t: t[0])
        return np.array([w for _, w in exp_list])

    weights = []
    for i in range(len(X_test_sub)):
        exp = explainer.explain_instance(
            data_row=X_test_sub.iloc[i].values,
            predict_fn=model.predict,
            num_features=X_test_sub.shape[1]
        )
        weights.append(_return_weights(exp))

    lime_weights = pd.DataFrame(weights, columns=X_test_sub.columns)

    # CSV
    csv_path = os.path.join(save_dir, f"{name}_weights.csv")
    lime_weights.to_csv(csv_path, index=False)

    # Global |weight| mean
    abs_mean_df = lime_weights.abs().mean().sort_values().reset_index()
    abs_mean_df.columns = ["feature", "abs_mean"]

    # Barh
    fig, ax = plt.subplots(figsize=(8, 4))
    yticks = range(len(abs_mean_df))
    ax.barh(yticks, abs_mean_df["abs_mean"])
    ax.set_yticks(yticks)
    ax.set_yticklabels(abs_mean_df["feature"], fontsize=12)
    ax.set_xlabel("Mean |LIME weight|", fontsize=14)
    ax.set_title("Importance globale (LIME) — test subset (n=10)", fontsize=14)
    plt.tight_layout()
    barh_out = os.path.join(save_dir, f"{name}_global.png")
    plt.savefig(barh_out, dpi=200)
    plt.close(fig)

    # Beeswarm
    fig, ax = plt.subplots(figsize=(8, 4))
    ylabels = abs_mean_df["feature"]
    yticks = range(len(ylabels))
    for i, feat in enumerate(ylabels):
        x_vals = lime_weights[feat]
        c_vals = X_test_sub[feat]
        ax.scatter(x_vals, [i]*len(x_vals), c=c_vals, cmap="bwr", edgecolors="k", alpha=0.8)
    ax.vlines(0, ymin=-1, ymax=len(ylabels), colors="k", linestyles="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_xlabel("LIME weight", fontsize=14)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Feature value", fontsize=12)
    plt.tight_layout()
    swarm_out = os.path.join(save_dir, f"{name}_beeswarm.png")
    plt.savefig(swarm_out, dpi=200)
    plt.close(fig)

    return {"weights_csv": csv_path, "barh_path": barh_out, "swarm_path": swarm_out}

# ============================
#  Appliquer toutes les méthodes XAI (MLP pipeline)
# ============================
def explain_mlp_pipeline(path_model: str, X_bg: pd.DataFrame, X_test_sub: pd.DataFrame,
                         y_test_sub: pd.Series, target: str):
    """
    Charge la pipeline MLP et applique SHAP (bg=100/test=10), PFI (test=10),
    PDP (bg=100), LIME (bg=100/test=10).
    """
    model = joblib.load(path_model)
    shap_dir, pfi_dir, pdp_dir, lime_dir = target_out_dirs(target)

    tag = f"{target}_bg{len(X_bg)}_test{len(X_test_sub)}"

    # SHAP
    shap_path = save_shap_summary(model, X_bg, X_test_sub, f"SHAP_{tag}", shap_dir)
    print(f"[{target}] SHAP  -> {shap_path}")

    # PFI
    pfi_path = save_pfi(model, X_test_sub, y_test_sub, f"PFI_{tag}", pfi_dir)
    print(f"[{target}] PFI   -> {pfi_path}")

    # PDP
    pdp_path = save_pdp(model, X_bg, FEATURES, f"PDP_{tag}", pdp_dir)
    print(f"[{target}] PDP   -> {pdp_path}")

    # LIME
    lime_out = save_lime(model, X_bg, X_test_sub, f"LIME_{tag}", lime_dir)
    print(f"[{target}] LIME  -> {lime_out}")

# ============================
#  Pipeline d’explication (N & S)
# ============================
def main():
    # Charger les données et splitter 
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURES]

    # N_sim
    yN = df["N_sim"]
    XN_train, XN_test, yN_train, yN_test = train_test_split(
        X, yN, test_size=TEST_SIZE, random_state=SEED
    )
    XN_bg, yN_bg, XN_test_sub, yN_test_sub = sample_train_test_subsets(
        XN_train, XN_test, yN_train, yN_test, BG_SIZE, TEST_SUB_SIZE, SEED
    )
    path_N = model_path_for_target("N_sim")
    if os.path.exists(path_N):
        explain_mlp_pipeline(path_N, XN_bg, XN_test_sub, yN_test_sub, "N_sim")
    else:
        print(f"[!] Modèle MLP introuvable pour N_sim : {path_N}")

    # S_sim
    yS = df["S_sim"]
    XS_train, XS_test, yS_train, yS_test = train_test_split(
        X, yS, test_size=TEST_SIZE, random_state=SEED
    )
    XS_bg, yS_bg, XS_test_sub, yS_test_sub = sample_train_test_subsets(
        XS_train, XS_test, yS_train, yS_test, BG_SIZE, TEST_SUB_SIZE, SEED
    )
    path_S = model_path_for_target("S_sim")
    if os.path.exists(path_S):
        explain_mlp_pipeline(path_S, XS_bg, XS_test_sub, yS_test_sub, "S_sim")
    else:
        print(f"[!] Modèle MLP introuvable pour S_sim : {path_S}")

if __name__ == "__main__":
    main()
