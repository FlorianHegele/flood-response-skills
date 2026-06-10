# -*- coding: utf-8 -*-
"""Skills crue exposés comme outils LangChain (couche anti-corruption).

Chaque skill du plugin est une CLI ``main.py`` qui résout un lieu (commune ou
code INSEE) et renvoie un JSON optimisé sur stdout. On l'enveloppe ici dans un
``@tool`` : l'outil shelle vers l'interpréteur des skills, capture le JSON et le
renvoie tel quel au LLM (le skill est déjà optimisé pour la décision).

Convention du dépôt respectée : on ne lève jamais sur un skill en échec ; on
renvoie un JSON d'erreur explicite (jamais de fallback silencieux), que le LLM
saura reformuler ou contourner.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

# Racine du dépôt = parent de orchestrateur/
REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / "skills"

# Interpréteur des skills : le .venv racine (auto-bootstrappé par les skills au
# 1er run) s'il existe, sinon l'interpréteur courant.
_venv_py = REPO / ".venv" / "bin" / "python"
SKILL_PY = str(_venv_py) if _venv_py.exists() else sys.executable


def _run_skill(skill: str, args: list[str], timeout: float = 180) -> str:
    """Exécute ``skills/<skill>/main.py`` et renvoie son JSON (string).

    Renvoie une chaîne JSON dans tous les cas :
    - stdout non vide → la sortie du skill (succès ou échec partiel) ;
    - sinon → un objet d'erreur reprenant stderr (échec total / fatal).
    """
    cmd = [SKILL_PY, str(SKILLS / skill / "main.py"), *args]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"error": f"timeout : le skill {skill} a dépassé {timeout:.0f}s"},
            ensure_ascii=False,
        )

    out = proc.stdout.strip()
    if out:
        # JSON déjà optimisé par le skill : on le relaie verbatim.
        return out

    # Pas de stdout : échec fatal, l'erreur est sur stderr (souvent du JSON).
    err = proc.stderr.strip()
    try:
        json.loads(err)
        return err
    except (ValueError, TypeError):
        return json.dumps(
            {"error": f"échec du skill {skill}", "detail": err or "(aucune sortie)"},
            ensure_ascii=False,
        )


# --- Simulation de crue (TEST) -------------------------------------------------
# Permet d'exercer la chaîne complète population→logistique hors période de crue.
# On NE court-circuite PAS le graphe : on superpose un scénario d'aléa sévère sur
# la VRAIE sortie d'alerte-crue (commune, coordonnées et stations réelles
# conservées), clairement étiqueté `_simulation`. Le pipeline tourne donc
# normalement : l'expert risque résume une crue cohérente, le classifieur la
# classe, le gate s'ouvre. Activé par la variable d'env SIMULER_VIGILANCE.
_SCENARIOS_VIGILANCE = {
    "vert": {"niveau": 1, "hauteur_mm": 400.0, "debit_ls": 1500.0,
             "pluie_mm": 0.0, "tendance": "stable", "label": "situation calme"},
    "jaune": {"niveau": 2, "hauteur_mm": 1500.0, "debit_ls": 40000.0,
              "pluie_mm": 40.0, "tendance": "hausse", "label": "montée modérée"},
    "orange": {"niveau": 3, "hauteur_mm": 3000.0, "debit_ls": 180000.0,
               "pluie_mm": 100.0, "tendance": "hausse rapide", "label": "crue importante"},
    "rouge": {"niveau": 4, "hauteur_mm": 5200.0, "debit_ls": 480000.0,
              "pluie_mm": 180.0, "tendance": "hausse rapide",
              "label": "crue majeure, épisode cévenol intense"},
}


def _simuler_vigilance(raw: str, couleur: str, commune: str) -> str:
    """Superpose un scénario d'aléa `couleur` sur la sortie réelle d'alerte-crue."""
    sc = _SCENARIOS_VIGILANCE.get(couleur.strip().lower())
    if sc is None:
        return raw
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError
    except (ValueError, TypeError):
        data = {"lieu": {"commune": commune}}

    data.setdefault("vigilance", {})
    data["vigilance"]["couleur"] = couleur
    data["vigilance"]["niveau"] = sc["niveau"]
    data["vigilance"].setdefault("troncon", "tronçon simulé")

    hydro = data.get("hydro") or {}
    stations = hydro.get("stations") or [
        {"station": "SIMU", "nom": "station simulée", "distance_km": 0.0}
    ]
    for st in stations:
        st["hauteur_mm"] = sc["hauteur_mm"]
        st["debit_ls"] = sc["debit_ls"]
        st["tendance"] = sc["tendance"]
    hydro["stations"] = stations
    data["hydro"] = hydro

    pluie = data.get("pluie") or {}
    pluie["cumul_prochaines_24h_mm"] = sc["pluie_mm"]
    pluie.setdefault("modele", "meteofrance_arome_france_hd")
    pluie.setdefault("unite", "mm")
    data["pluie"] = pluie

    data["_simulation"] = (
        f"DONNÉE SIMULÉE (SIMULER_VIGILANCE={couleur}) : {sc['label']}. "
        "Vigicrues / Hub'Eau / AROME réels NON reflétés — usage test uniquement."
    )
    return json.dumps(data, ensure_ascii=False)


@tool
def alerte_crue(commune: str) -> str:
    """Risque de crue d'une commune française : couleur de vigilance Vigicrues,
    hauteur/débit temps réel des stations proches, et prévision de pluie (AROME).
    `commune` = nom ou code INSEE (ex. "Alès" ou "30007")."""
    raw = _run_skill("alerte-crue", ["--commune", commune])
    sim = os.getenv("SIMULER_VIGILANCE")
    return _simuler_vigilance(raw, sim, commune) if sim else raw


@tool
def demographie_iris(commune: str) -> str:
    """Démographie d'une commune au niveau quartier (IRIS) : population, ménages,
    familles monoparentales — pour estimer le nombre de personnes à évacuer/héberger.
    `commune` = nom ou code INSEE."""
    return _run_skill("demographie-iris", ["--commune", commune, "--top", "10"])


@tool
def vulnerabilite_bpe(commune: str) -> str:
    """Équipements sensibles d'une commune : écoles (enfants à évacuer) et
    établissements de santé (hôpitaux, maternités, dialyse — continuité des soins).
    `commune` = nom ou code INSEE."""
    return _run_skill("vulnerabilite-bpe", ["--commune", commune, "--top", "30"])


@tool
def accessibilite_routes(commune: str) -> str:
    """Ouvrages routiers vulnérables autour d'une commune (gués, ponts, tunnels,
    passages inférieurs) susceptibles d'être coupés par une crue — pour planifier
    accès et itinéraires d'évacuation. `commune` = nom ou code INSEE."""
    return _run_skill(
        "accessibilite-routes",
        ["--commune", commune, "--radius-m", "2000", "--limit", "50"],
    )


@tool
def logistique_hebergement(commune: str) -> str:
    """Sites d'hébergement d'urgence autour d'une commune (hôtels, gymnases, écoles,
    salles communales) avec une capacité de couchage estimée et étiquetée — pour
    mettre les sinistrés à l'abri. `commune` = nom ou code INSEE."""
    return _run_skill(
        "logistique-hebergement",
        ["--commune", commune, "--radius-m", "3000", "--limit", "50"],
    )
