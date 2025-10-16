# /data.py
import numpy as np
import pandas as pd
from pathlib import Path

# Projet
from benchmark.io import DATA        
from benchmark.config import SEED       

# ----------------------------
# Paramètres ajustables
# ----------------------------
INPUT_CSV   = "synthetic_dataset_filtered.csv"                 # BDD d'entrée
OUTPUT_CSV  = "perturbed_data.csv"      # BDD de sortie

ALPHA       = 0.8   # 0<ALPHA<1  -> poids de ppH2 dans la variable "partiellement proportionnelle"
NOISE_SCALE = 0.1  # échelle du bruit en "écarts-types" de ppH2 (1.0 = même ordre de grandeur)

# ----------------------------
# Main
# ----------------------------
def main():
    rng = np.random.default_rng(SEED)

    in_path  = DATA / INPUT_CSV
    out_path = DATA / OUTPUT_CSV

    df = pd.read_csv(in_path)

    if "ppH2" not in df.columns:
        raise KeyError("La colonne 'ppH2' est absente de la BDD.")

    # --- Variable partiellement proportionnelle à ppH2 ---
    # Bruit centré, d'écart-type proportionnel à std(ppH2)
    std_ppH2 = float(df["ppH2"].std(ddof=0)) if df["ppH2"].std(ddof=0) > 0 else 1.0
    noise = rng.normal(loc=0.0, scale=NOISE_SCALE * std_ppH2, size=len(df))

    # Mélange linéaire: α*ppH2 + (1-α)*noise
    ppH2_partial = ALPHA * df["ppH2"].to_numpy() + (1.0 - ALPHA) * noise

    df["aux_1"] = ppH2_partial

    # --- Variable indépendante uniformément entre 0 et 4 ---
    df["aux_2"] = rng.uniform(0.0, 4.0, size=len(df))

    # Sauvegarde
    df.to_csv(out_path, index=False)
    print(f"[OK] Colonnes ajoutées: 'aux_1', 'aux_2'")
    print(f"[→] Fichier sauvegardé: {out_path}")

if __name__ == "__main__":
    main()
