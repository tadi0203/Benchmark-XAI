# data_generation/bdd_synthetique.py

import pandas as pd
import numpy as np
import itertools
from benchmark.io import DATA, ensure_dir
from benchmark.config import FEATURES  # ["T_k","T05","TMP","ppH2","ppH2S","ppNH3","Resines_chg","t","N0","S0"]

from benchmark.physics import simulate_row

REAL_DATA_PATH = DATA / "simulated_results.csv"
OUTPUT_PATH    = DATA / "synthetic_dataset_filtered.csv"
def main():
    #=== Étape 0 :  Lecture des données  
    if not REAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Introuvable: {REAL_DATA_PATH}")
    df_real = pd.read_csv(REAL_DATA_PATH)

    # === Variables d'entrée ===
    input_vars = FEATURES

    # === Étape 1 : Extraire min, mean, max pour chaque variable
    value_options = {
        var: [df_real[var].min(), df_real[var].mean(), df_real[var].max()]
        for var in input_vars
    }

    # === Étape 2 : Générer toutes les combinaisons (3^10 = 59 049 lignes)
    all_combinations = list(itertools.product(*value_options.values()))
    print(f"[i] Nombre total de combinaisons générées : {len(all_combinations)}")

    # === Étape 3 : Créer DataFrame synthétique
    synthetic_df = pd.DataFrame(all_combinations, columns=input_vars)

    # === Étape 4 : Appliquer simulateur physique
    N_results = []
    S_results = []

    for i, row in synthetic_df.iterrows():
        sim_out = simulate_row(row)
        N_results.append(sim_out[0])
        S_results.append(sim_out[1])
        if i % 1000 == 0:
            print(f"  → {i} lignes traitées...")

    # Ajouter les sorties
    synthetic_df["N_sim"] = N_results
    synthetic_df["S_sim"] = S_results

    # === Étape 5 : Filtrage
    filtered_df = synthetic_df[
        (synthetic_df["N_sim"] >= 5) & (synthetic_df["N_sim"] <= 2000) &
        (synthetic_df["S_sim"] >= 5) & (synthetic_df["S_sim"] <= 2000)
    ]

    print(f"[OK] Nombre de lignes après filtrage : {len(filtered_df)}")
    
    ensure_dir(DATA)
    # === Étape 6 : Sauvegarde
    filtered_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Base synthétique filtrée enregistrée dans : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()