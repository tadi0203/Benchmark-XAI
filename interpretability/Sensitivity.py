# interpretability/sensitivity.py
# Calcul de delta des variables d'entées pour avoir 10 ppm sur la sortie N/S
# ============================
#  BIBLIOTHÈQUES
# ============================
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

#  Chemins et config 
from benchmark.physics import simulate_row  # Fonction de simulation du modèle HDN/HDS
from benchmark.io import  XAI,DATA

# ----------------------------
#  Chemins E/S 
# ----------------------------
OLD_DATA_CSV = DATA / "simulated_results.csv"  
EXPLAIN_BASE_PATH      = XAI / "deltas"

# ============================
#  Fonction de calcul de variation nécessaire (Δ)
# ============================
def compute_required_delta_optimized(df_row, target='N', target_change=10, initial_step=0.1, max_iter=500):
    """
    Calcule pour chaque variable d'entrée (input_cols) le delta minimum à appliquer pour
    obtenir un changement d'au moins +target_change ppm dans la cible choisie (N ou S).
    Les variables sont en %mass sauf en interne pour simulate_row (qui utilise ppm).
    """
    baseline = simulate_row(convert_row_for_simulation(df_row))
    target_idx = 0 if target == 'N' else 1
    base_val = baseline[target_idx]
    results = {}

    for col in input_cols:
        step = initial_step
        delta = step
        iter_count = 0
        found = False

        while iter_count < max_iter:
            df_mod = df_row.copy()
            df_mod[col] = df_row[col] + delta
            new_val = simulate_row(convert_row_for_simulation(df_mod))[target_idx]
            variation = new_val - base_val

            if abs(variation) >= target_change:
                results[col] = delta
                found = True
                break
            step *= 2
            delta += step
            iter_count += 1

        if not found:
            results[col] = None

    return results

# ============================
#  Fonction utilitaire : conversion vers ppm pour simulateur
# ============================
def convert_row_for_simulation(row):
    """Convertit %mass → ppm uniquement pour Resines_chg (le simulateur attend ppm)"""
    row_sim = row.copy()
    row_sim['Resines_chg'] = row_sim['Resines_chg'] * 10000  # % → ppm
    return row_sim

# ============================
#  Affichage des résultats sous forme de barplot
# ============================
def plot_deltas(delta_dict, target, row_full):
    """
    Génère un histogramme horizontal des deltas calculés.
    """
    labels, values, colors = [], [], []

    for var, delta in delta_dict.items():
        if delta is not None:
            sign_label = '+' if expected_signs[var] > 0 else '−'
            label = f"{var} ({sign_label}) [{units[var]}]"
            labels.append(label)
            values.append(delta)
            colors.append('green' if expected_signs[var] > 0 else 'red')

    plt.figure(figsize=(10, 5))
    bars = plt.barh(labels, values, color=colors)
    plt.axvline(0, color='black', linestyle='--')

    T_k = row_full["T_k"]
    ppH2 = row_full["ppH2"]
    N0 = row_full["N0"]
    Resines_chg_pct = row_full["Resines_chg"]  # déjà en %mass
    sim_val = row_full["N_sim"] if target == "N" else row_full["S_sim"]

    title_str = (
        f"Δ nécessaire pour augmenter {target} de +10 ppm\n"
        f"T_k={T_k:.1f} K, ppH2={ppH2:.1f} bar, Resines_chg={Resines_chg_pct:.4f} %, "
        f"N0={N0:.1f} ppm, {target}_sim={sim_val:.1f} ppm"
    )

    plt.title(title_str)
    plt.xlabel("Delta appliqué")

    for bar, val in zip(bars, values):
        plt.text(val, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                 va='center', ha='left' if val > 0 else 'right')

    plt.tight_layout()
    os.makedirs(EXPLAIN_BASE_PATH, exist_ok=True)
    plt.savefig(os.path.join(EXPLAIN_BASE_PATH, f"Deltas_{target}.png"))
    plt.close()

# ============================
#  Script principal
# ============================
if __name__ == "__main__":

    #  Chargement des données
    df = pd.read_csv(OLD_DATA_CSV)

    # Conversion de Resines_chg en % massique pour traitement (diviser par 10000)
    df["Resines_chg"] = df["Resines_chg"] / 10000

    #  Filtrage de lignes représentatives (autour de 50 ppm)
    filtered_df_N = df[df['N_sim'].between(45, 55)]
    filtered_df_S = df[df['S_sim'].between(45, 55)]

    row_N = filtered_df_N.iloc[0]
    row_S = filtered_df_S.iloc[0]

    #  Paramètres globaux
    target_change = 10  # Objectif : +10 ppm
    input_cols = ['T_k', 'ppH2', 'N0', 'Resines_chg']
    units = {'T_k': 'K', 'ppH2': 'bar', 'N0': 'ppm', 'Resines_chg': '% mass'}
    expected_signs = {'T_k': -1, 'ppH2': -1, 'N0': +1, 'Resines_chg': +1}  # Signe attendu de l'effet

    #  Calcul et affichage pour N
    deltas_N = compute_required_delta_optimized(row_N, target='N', target_change=target_change)
    plot_deltas(deltas_N, target='N', row_full=row_N)

    #  Calcul et affichage pour S
    deltas_S = compute_required_delta_optimized(row_S, target='S', target_change=target_change)
    plot_deltas(deltas_S, target='S', row_full=row_S)
