# -*- coding: utf-8 -*-
"""Orchestrateur LangGraph (cf. cours, chap. 6 — ``StateGraph``).

On encode le flux de décision en crue plutôt que de le laisser à un superviseur
LLM libre, parce que les skills sont couplés :

1. GATE : la vigilance (alerte-crue) commande le reste. Vigilance verte/inconnue
   → on saute directement à la synthèse (inutile de mobiliser population/routes/
   hébergement, donc moins d'appels API/LLM).
2. DÉPENDANCE : le besoin en hébergement dépend du nombre d'évacués. L'état
   partagé transporte le digest population jusqu'au nœud logistique.
3. SYNTHÈSE : un nœud final compose la note de décision.

    START → noeud_risque ──[routeur]── vert/inconnu ─→ noeud_synthese → END
                            │ jaune/orange/rouge
                            └→ noeud_population → noeud_logistique → noeud_synthese → END
"""

from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from experts import make_experts
from llm import get_llm


class Etat(TypedDict, total=False):
    """État partagé qui circule dans le graphe."""

    lieu: str            # commune ou code INSEE
    risque: str          # digest de l'expert risque
    couleur: str         # couleur de vigilance extraite (pour le gate)
    mobiliser: bool      # faut-il déclencher la chaîne d'évacuation ?
    population: str      # digest de l'expert population
    logistique: str      # digest de l'expert logistique
    note: str            # note de décision finale


class NiveauRisque(BaseModel):
    """Classification structurée du digest risque (chap. 2) pour piloter le gate."""

    couleur: Literal["vert", "jaune", "orange", "rouge", "inconnu"] = Field(
        description="Couleur de vigilance crue dominante d'après le résumé."
    )
    mobiliser: bool = Field(
        description="Vrai s'il faut déclencher l'évacuation (vigilance jaune, "
        "orange ou rouge, ou montée des eaux avérée)."
    )


def build_graph(recursion_limit: int = 50):
    """Construit et compile le graphe d'orchestration."""

    expert_risque, expert_population, expert_logistique = make_experts()
    classifieur = get_llm().with_structured_output(NiveauRisque)

    def _digest(agent, consigne: str) -> str:
        """Invoque un sous-agent et renvoie le texte de sa réponse finale."""
        res = agent.invoke({"messages": [{"role": "user", "content": consigne}]})
        return res["messages"][-1].content

    def noeud_risque(etat: Etat) -> Etat:
        lieu = etat["lieu"]
        digest = _digest(expert_risque, f"Lieu : {lieu}. Évalue le risque de crue.")
        niveau = classifieur.invoke(
            "Classe ce résumé de risque de crue :\n" + digest
        )
        return {
            "risque": digest,
            "couleur": niveau.couleur,
            "mobiliser": niveau.mobiliser,
        }

    def routeur(etat: Etat) -> Literal["noeud_population", "noeud_synthese"]:
        # Routeur idempotent et rapide : lit l'état, n'appelle pas de LLM (slide 85).
        return "noeud_population" if etat.get("mobiliser") else "noeud_synthese"

    def noeud_population(etat: Etat) -> Etat:
        lieu = etat["lieu"]
        digest = _digest(
            expert_population,
            f"Lieu : {lieu}. Estime la population à évacuer et les sites sensibles.",
        )
        return {"population": digest}

    def noeud_logistique(etat: Etat) -> Etat:
        lieu = etat["lieu"]
        # Dépendance évacués → couchages : on injecte le digest population.
        consigne = (
            f"Lieu : {lieu}. Besoin d'évacuation estimé (à dimensionner) :\n"
            f"{etat.get('population', '(non précisé)')}\n\n"
            "Identifie les routes à risque de coupure et les hébergements ; "
            "compare la capacité de couchage au besoin ci-dessus."
        )
        return {"logistique": _digest(expert_logistique, consigne)}

    def noeud_synthese(etat: Etat) -> Etat:
        llm = get_llm()
        contexte = (
            f"LIEU : {etat.get('lieu')}\n"
            f"VIGILANCE : {etat.get('couleur', 'inconnu')}\n\n"
            f"--- RISQUE ---\n{etat.get('risque', 'non évalué')}\n\n"
            f"--- POPULATION ---\n{etat.get('population', 'non mobilisée (pas de risque avéré)')}\n\n"
            f"--- LOGISTIQUE ---\n{etat.get('logistique', 'non mobilisée (pas de risque avéré)')}"
        )
        msg = [
            {
                "role": "system",
                "content": (
                    "Tu es le coordinateur de crise. À partir des constats des experts, "
                    "rédige une NOTE DE DÉCISION concise et actionnable, structurée en : "
                    "1) Situation et niveau de danger ; 2) Qui évacuer / protéger (avec "
                    "ordres de grandeur) ; 3) Accès et itinéraires à surveiller ; "
                    "4) Hébergement (capacité vs besoin) ; 5) Recommandation immédiate. "
                    "Si la vigilance est verte/inconnue, dis-le et recommande la veille "
                    "plutôt que la mobilisation. N'invente aucun chiffre absent des constats."
                ),
            },
            {"role": "user", "content": contexte},
        ]
        return {"note": llm.invoke(msg).content}

    g = StateGraph(Etat)
    g.add_node("noeud_risque", noeud_risque)
    g.add_node("noeud_population", noeud_population)
    g.add_node("noeud_logistique", noeud_logistique)
    g.add_node("noeud_synthese", noeud_synthese)

    g.add_edge(START, "noeud_risque")
    g.add_conditional_edges("noeud_risque", routeur)
    g.add_edge("noeud_population", "noeud_logistique")
    g.add_edge("noeud_logistique", "noeud_synthese")
    g.add_edge("noeud_synthese", END)

    return g.compile().with_config(recursion_limit=recursion_limit)
