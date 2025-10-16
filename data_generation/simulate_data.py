# data_generation/simulate_data.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns

from benchmark.io import DATA, ensure_dir
from benchmark.physics import system  # système d’EDO

# =======================
#  Chargement & nettoyage
# =======================
def load_and_clean_data(path_xls: Path | str) -> pd.DataFrame:
    """
    Charge le fichier Excel, nettoie les données, convertit les colonnes numériques
    et applique les transformations physiques nécessaires à la simulation.
    """
    df = pd.read_excel(path_xls)

    # Colonnes numériques à convertir (adapte si besoin)
    NUM_COLS = [
        "T", "VVH", "t", "ppH2", "Resines_chg", "N0", "S0",
        "T05", "T50", "T70", "T80", "T95",
        "Azote_liqTot", "Soufre_liqTot", "ppH2S", "ppNH3",
        "N_eq", "S_eq", "P_tot",
    ]

    # Convertir en numérique (gère virgules décimales, ND, tirets…)
    def to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df[cols] = (
            df[cols]
            .astype(str)
            .replace({",": ".", "—": np.nan, "-": np.nan, "ND": np.nan, "": np.nan}, regex=True)
            .apply(pd.to_numeric, errors="coerce")
        )
        return df

    df = to_float(df, NUM_COLS)

    # Filtrage : garder lignes proches de l'équilibre (N0 ≈ N_eq et S0 ≈ S_eq)
    RTOL, ATOL = 1e-4, 1e-4
    mask_N = np.isclose(df["N0"], df["N_eq"], rtol=RTOL, atol=ATOL)
    mask_S = np.isclose(df["S0"], df["S_eq"], rtol=RTOL, atol=ATOL)
    df = df[mask_N & mask_S].reset_index(drop=True)

    # Conversions d’unités / features dérivées
    df["Soufre_liqTot"] *= 1e4       # → ppm
    df["S0"] *= 1e4                  # → ppm
    df["Resines_chg"] *= 1e4         # → ppm
    df["T_k"] = df["T"] + 273.15     # °C → K
    df["TMP"] = (df["T05"] + 2*df["T50"] + 4*df["T95"]) / 7  # moyenne pondérée
    df["t"] /= 3600                  # secondes → heures

    return df

# =======================
#  Simulation des réactions
# =======================
def simulate_reactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique la simulation physique (solve_ivp) pour chaque ligne.
    Retourne df fusionné avec colonnes N_sim / S_sim + N_exp / S_exp.
    """
    results = []
    for _, r in df.iterrows():
        args = (r.T_k, r.T05, r.TMP, r.ppH2, r.ppH2S, r.ppNH3, r.N0, r.S0, r.Resines_chg)
        sol = solve_ivp(
            system,
            (0, r.t),
            [r.N0, r.S0],
            args=args,
            method="LSODA",
            t_eval=[r.t],
        )
        N_sim, S_sim = sol.y[:, 0]

        results.append({
            "Nom_bilan": r.Nom_bilan,
            "Nom_charge": r.Nom_charge,
            "N_sim": float(N_sim),
            "S_sim": float(S_sim),
            "N_exp": float(r.Azote_liqTot),
            "S_exp": float(r.Soufre_liqTot),
        })

    cmp = pd.DataFrame(results).drop(columns=["Nom_charge"])
    return pd.merge(df, cmp, on="Nom_bilan")

# =======================
#  Sauvegarde CSV
# =======================
def save_results(df_sim: pd.DataFrame, path: Path | str = DATA / "simulated_results.csv") -> None:
    ensure_dir(DATA)
    pd.DataFrame(df_sim).to_csv(path, index=False)
    print(f"[✓] Résultats enregistrés : {path}")

# =======================
#  Graphe de parité
# =======================
def plot_parity(df_sim: pd.DataFrame, save_path: Path | str | None = DATA / "parity_plot.png") -> None:
    """
    Graphe de parité (N_exp vs N_sim et S_exp vs S_sim).
    Si save_path est None → affichage, sinon sauvegarde.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # — Parité Azote
    sns.scatterplot(data=df_sim, x="N_exp", y="N_sim", hue="Nom_charge", ax=ax[0])
    limN = [0, max(df_sim["N_exp"].max(), df_sim["N_sim"].max()) * 1.1]
    ax[0].plot(limN, limN, "k-", label="y = x")
    ax[0].plot(limN, [1.15*x for x in limN], "k--", alpha=.6)
    ax[0].plot(limN, [0.85*x for x in limN], "k--", alpha=.6)
    ax[0].plot(limN, [1.30*x for x in limN], "k:", alpha=.6)
    ax[0].plot(limN, [0.70*x for x in limN], "k:", alpha=.6)
    ax[0].set_xlim(limN); ax[0].set_ylim(limN)
    ax[0].set_title("Parité Azote – toutes charges")
    ax[0].set_xlabel("Azote exp (ppm)"); ax[0].set_ylabel("Azote sim (ppm)")

    # — Parité Soufre
    sns.scatterplot(data=df_sim, x="S_exp", y="S_sim", hue="Nom_charge", ax=ax[1])
    limS = [0, max(df_sim["S_exp"].max(), df_sim["S_sim"].max()) * 1.1]
    ax[1].plot(limS, limS, "k-", label="y = x")
    ax[1].plot(limS, [1.15*x for x in limS], "k--", alpha=.6)
    ax[1].plot(limS, [0.85*x for x in limS], "k--", alpha=.6)
    ax[1].plot(limS, [1.30*x for x in limS], "k:", alpha=.6)
    ax[1].plot(limS, [0.70*x for x in limS], "k:", alpha=.6)
    ax[1].set_xlim(limS); ax[1].set_ylim(limS)
    ax[1].set_title("Parité Soufre – toutes charges")
    ax[1].set_xlabel("Soufre exp (ppm)"); ax[1].set_ylabel("Soufre sim (ppm)")

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"[✓] Figure enregistrée : {save_path}")
    else:
        plt.show()

# =======================
#  Pipeline principal
# =======================
def main() -> None:
    # Chemin vers la base expérimentale (XLSX) à la racine data/
    xls_path = DATA / "real bdd modif.xlsx"  # adapte le nom exact si besoin

    # 1) Chargement + nettoyage
    df = load_and_clean_data(xls_path)

    # 2) Simulation (solve_ivp pour chaque ligne)
    df_sim = simulate_reactions(df)

    # 3) Sauvegarde CSV + Parity Plot
    save_results(df_sim, DATA / "simulated_results.csv")
    plot_parity(df_sim, DATA / "parity_plot.png")

if __name__ == "__main__":
    main()
