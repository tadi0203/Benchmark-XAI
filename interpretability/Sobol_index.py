# interpretability/Sobol_index.py
# calcul l'indice de sobol pour les variable d'entrée sur le modele physique
# ============================
#  BIBLIOTHÈQUES
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
import os

#  Chemins et config 
from benchmark.physics import simulate_row
from benchmark.io import  XAI,DATA
from benchmark.config import  FEATURES   

# ----------------------------
#  Chemins E/S 
# ----------------------------
OLD_DATA_CSV = DATA / "simulated_results.csv"  
EXPLAIN_BASE_PATH      = XAI / "sobol_indices"




# === Constantes globales ===
N_SAMPLES = 128  # Nombre de points pour l'échantillonnage de Saltelli
# === Variables d'entrée du modèle physique ===
input_vars = FEATURES

units = {'T_k': 'K','T05': 'K' ,'TMP': 'K','ppH2': 'bar','ppH2S':'bar','ppNH3':'bar', 'Resines_chg': 'ppm','N0': 'ppm','S0': 'ppm','t':'h'}


def define_problem(df):
    """Définit le dictionnaire `problem` requis par SALib à partir des min/max des variables dans le DataFrame."""
    bounds = [[df[col].min(), df[col].max()] for col in input_vars]
    return {
        'num_vars': len(input_vars),
        'names': input_vars,
        'bounds': bounds
    }


def run_physical_model(X, row_template, target='N'):
    """
    Exécute le simulateur `simulate_row` en perturbant les variables d'entrée autour d'une ligne de référence.
    Retourne un vecteur de sorties (N ou S) pour les points de l'échantillon SALib.
    """
    results = []
    idx = 0 if target == 'N' else 1
    for row in X:
        row_sim = row_template.copy()
        for i, var in enumerate(input_vars):
            row_sim[var] = row[i]
        out = simulate_row(row_sim)
        results.append(out[idx])
    return np.array(results)


def average_sobol_indices(list_Si, problem):
    """
    Calcule la moyenne des indices de Sobol (S1, ST) à partir d'une liste de dictionnaires Si.
    Ignore les cas invalides ou mal formatés.
    """
    keys = ['S1', 'ST']
    avg = {}
    valid_Si = [si for si in list_Si if all(k in si and isinstance(si[k], np.ndarray) for k in keys)]

    if len(valid_Si) == 0:
        raise ValueError("Aucun résultat Sobol valide trouvé pour faire la moyenne.")

    for k in keys:
        values = np.array([si[k] for si in valid_Si])
        avg[k] = np.nanmean(values, axis=0)

    avg['names'] = problem['names']
    return avg


def plot_sobol_indices(Si_avg, target, save_path):
    """
    Génère un graphique des indices de Sobol moyens (S1 et ST) pour une variable cible donnée (N ou S).
    Enregistre le graphique dans le dossier fourni.
    """
    S1 = Si_avg['S1']
    ST = Si_avg['ST']
    features = Si_avg['names']
    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, S1, width, label='1er ordre', color='skyblue')
    ax.bar(x + width/2, ST, width, label='Total', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.set_ylabel("Indice de Sobol")
    ax.set_title(f"Analyse de Sobol moyenne – Variable cible : {target}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sobol_avg_{target}.png"))
    plt.close()


def compute_average_sobol(df, target='N'):
    """
    Applique l’analyse de Sobol sur chaque ligne du dataset, puis fait la moyenne des indices obtenus.
    La sortie Y est soit N, soit S selon `target`.
    """
    print(f"[…] Lancement Sobol pour la variable cible : {target}")

    problem = define_problem(df)
    X = saltelli.sample(problem, N_SAMPLES, calc_second_order=False)
    sobol_results = []

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            Y = run_physical_model(X, row, target)
            Si = sobol.analyze(problem, Y, calc_second_order=False)
            sobol_results.append(Si)

            if i % 50 == 0:
                print(f"  → Ligne {i}/{len(df)} traitée.")
        except Exception as e:
            print(f"[!] Erreur à la ligne {i}: {e}")
            continue

    Si_avg = average_sobol_indices(sobol_results, problem)
    return Si_avg


def main():
    """
    Point d'entrée principal : lit les données, calcule les indices de Sobol pour N et S,
    et génère les graphiques dans le dossier de sortie.
    """
    os.makedirs(EXPLAIN_BASE_PATH, exist_ok=True)
    df = pd.read_csv(OLD_DATA_CSV)

    print("[...] Moyenne des indices de Sobol pour N (TOUTES les lignes)")
    Si_avg_N = compute_average_sobol(df, target='N')

    plot_sobol_indices(Si_avg_N, 'N', EXPLAIN_BASE_PATH)

    Si_avg_S = compute_average_sobol(df, target='S')

    plot_sobol_indices(Si_avg_S, 'S', EXPLAIN_BASE_PATH)

    print("[✓] Graphes moyens enregistrés.")



if __name__ == "__main__":
    main()
