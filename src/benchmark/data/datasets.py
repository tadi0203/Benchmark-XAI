# src/benchmark/data/datasets.py
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import pandas as pd

from ..io import DATA
from ..config import FEATURES 

@dataclass
class DatasetConf:
    csv: str
    features: list[str]

DATASETS: dict[str, DatasetConf] = {
    "old_bdd": DatasetConf(csv="simulated_results.csv",           features=FEATURES),
    "new_bdd": DatasetConf(csv="synthetic_dataset_filtered.csv",  features=FEATURES),
}

def get_df(name: str) -> pd.DataFrame:
    if name not in DATASETS:
        raise KeyError(f"Dataset inconnu: {name}. Dispo: {list(DATASETS)}")
    p = DATA / DATASETS[name].csv
    if not p.exists():
        raise FileNotFoundError(f"Introuvable: {p}")
    return pd.read_csv(p)

def load_xy(name: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    conf = DATASETS[name]
    df = get_df(name)
    if target not in df.columns:
        raise KeyError(f"Colonne cible '{target}' absente de {conf.csv}. Colonnes: {list(df.columns)}")
    X = df[conf.features].copy()
    y = df[target].copy()
    return X, y
