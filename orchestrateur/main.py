# -*- coding: utf-8 -*-
"""CLI de l'orchestrateur crue : commune → note de décision.

Exemples :
    uv run python main.py --commune "Alès"
    uv run python main.py --commune 30007 --json
    LLM_BACKEND=ollama uv run python main.py --commune "Nîmes"
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Orchestrateur LangGraph : produit une note de décision en crue "
        "pour une commune française, en consultant 3 sous-agents experts."
    )
    p.add_argument("--commune", help="Nom ou code INSEE (ex. \"Alès\" ou 30007)")
    p.add_argument(
        "--backend",
        choices=["mistral", "ollama", "anthropic"],
        help="Override de LLM_BACKEND pour ce run.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Affiche l'état complet du graphe (debug) au lieu de la seule note.",
    )
    p.add_argument(
        "--simuler-vigilance",
        choices=["vert", "jaune", "orange", "rouge"],
        help="TEST : superpose un scénario d'aléa de cette couleur sur la vraie "
        "sortie d'alerte-crue (commune/stations réelles conservées) pour exercer "
        "la chaîne complète population→logistique→synthèse hors période de crue.",
    )
    return p


def main(argv=None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend

    if args.simuler_vigilance:
        # Lu par skills_tools.alerte_crue (même process) pour injecter le scénario.
        os.environ["SIMULER_VIGILANCE"] = args.simuler_vigilance

    if not args.commune:
        print(
            json.dumps(
                {"error": "fournir --commune (nom ou code INSEE)"}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 2

    # Import tardif : ne charge LangChain qu'après le parsing des arguments.
    from orchestrateur import build_graph

    graph = build_graph()
    etat = graph.invoke({"lieu": args.commune})

    if args.json:
        print(json.dumps(etat, ensure_ascii=False, indent=2))
    else:
        print(etat.get("note", "(aucune note produite)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
