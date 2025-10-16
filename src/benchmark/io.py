from pathlib import Path

# Racine du projet (dossier Benchmark/)
ROOT = Path(__file__).resolve().parents[2]

# Dossiers principaux
DATA   = ROOT / "data"          # pour stocker CSV/XLSX
MODELS = ROOT / "models"        # pour sauvegarder/charger les modèles

# Dossiers de sortie pour analyses
XAI    = ROOT / "interpretability"   # graphiques et résultats XAI
ROBUST = ROOT / "evaluation"         # graphiques de robustesse

def ensure_dir(p: Path) -> None:
    """Créer un dossier s'il n'existe pas (équivalent mkdir -p)."""
    p.mkdir(parents=True, exist_ok=True)
