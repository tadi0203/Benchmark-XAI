# full_env_setup.ps1
# Ce script prépare un environnement Python/Poetry pour le projet Benchmark
# Auteur : Mohtadi HAMMAMI

Write-Host "Démarrage de la configuration de l'environnement Benchmark..."

# Étape 1 : Configuration proxy 
$env:http_proxy = "http://proxy.ens-lyon.fr:3128"
$env:https_proxy = "http://proxy.ens-lyon.fr:3128"
Write-Host "Proxy configuré"

# Étape 2 : Installation de Scoop 
if (-not (Get-Command scoop -ErrorAction SilentlyContinue)) {
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    iwr -useb get.scoop.sh | iex
    Write-Host "Scoop installé avec succès"
} else {
    Write-Host "Scoop déjà installé"
}

# Étape 3 : Installation de Python 3.11 via Scoop
scoop install python311
scoop reset python311
Write-Host "Python 3.11 installé et activé via Scoop"

# Étape 4 : Installation de Poetry 
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "Installation de Poetry via pipx..."
    scoop install pipx
    pipx ensurepath
    pipx install poetry
    Write-Host "Poetry installé"
} else {
    Write-Host "Poetry déjà installé"
}

# Étape 5 : Chemin vers l'exécutable Python (dans mon cas)
$pythonPath = "$env:USERPROFILE\scoop\apps\python311\current\python.exe"

# Étape 6 : Aller dans le dossier du projet
$projectPath = "$PSScriptRoot"  # dossier courant où se trouve ce script
Set-Location $projectPath

# Étape 7 : Initialisation du projet Poetry (si pyproject.toml absent)
if (-not (Test-Path "$projectPath\pyproject.toml")) {
    poetry init --name benchmark --description "Benchmark ML" --author "HAMMAMI Mohtadi" --python "^3.11" --no-interaction
    Write-Host "Fichier pyproject.toml généré"
}

# Étape 8 : Utiliser le bon interpréteur Python
poetry env use $pythonPath
Write-Host "Environnement virtuel configuré avec Python 3.11.9"

# Étape 9 : Ajouter les dépendances du projet
poetry add pandas@^2.2.2 numpy@^1.26.4 scikit-learn@^1.4.2 matplotlib@^3.8.4 seaborn@^0.13.2 shap@^0.48.0 lime@^0.2.0.1 joblib@^1.4.2 openpyxl@^3.1.2
Write-Host "Dépendances installées"

# Étape 10 : Installer les dépendances à partir du lockfile
poetry lock
poetry install --no-root
Write-Host "Projet installé "

# Étape 11 : Activer l’environnement 
$activateScript = "$projectPath\.venv\Scripts\Activate.ps1"
Write-Host "Exécution terminée. Pour activer l'environnement, exécute :"
Write-Host ""
Write-Host "    & `"$activateScript`""
