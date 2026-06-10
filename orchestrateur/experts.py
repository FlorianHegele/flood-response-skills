# -*- coding: utf-8 -*-
"""Les 3 sous-agents experts (cf. cours, chap. 4 — ``create_agent``).

Chaque expert est un agent autonome (boucle LLM ↔ outils) qui regroupe des
skills liés et raisonne dessus :
- Risque      : alerte_crue
- Population   : demographie_iris + vulnerabilite_bpe (croise habitants et sites sensibles)
- Logistique  : accessibilite_routes + logistique_hebergement (compare besoin et capacité)

Les system_prompts restent courts et spécifiques (slides 45/57/89) : un expert
ciblé répond mieux qu'un méga-prompt généraliste.
"""

from langchain.agents import create_agent

from llm import get_llm
from skills_tools import (
    accessibilite_routes,
    alerte_crue,
    demographie_iris,
    logistique_hebergement,
    vulnerabilite_bpe,
)


def make_experts():
    """Construit les 3 sous-agents experts (un appel get_llm() par expert)."""

    expert_risque = create_agent(
        model=get_llm(),
        tools=[alerte_crue],
        system_prompt=(
            "Tu es hydrologue de crise. Appelle l'outil alerte_crue pour le lieu "
            "indiqué, puis résume en 3 lignes maximum : la couleur de vigilance "
            "Vigicrues, le niveau d'eau / la tendance des stations, et la pluie à "
            "venir. Sois factuel ; si une source est indisponible, dis-le sans inventer."
        ),
    )

    expert_population = create_agent(
        model=get_llm(),
        tools=[demographie_iris, vulnerabilite_bpe],
        system_prompt=(
            "Tu es démographe d'urgence. Pour le lieu indiqué, appelle "
            "demographie_iris ET vulnerabilite_bpe, puis croise-les. Indique "
            "EXPLICITEMENT un ordre de grandeur du NOMBRE de personnes à évacuer "
            "(population concernée), les quartiers vulnérables (familles "
            "monoparentales), et les sites sensibles : écoles (enfants) et "
            "établissements de santé (continuité des soins). Reste concis."
        ),
    )

    expert_logistique = create_agent(
        model=get_llm(),
        tools=[accessibilite_routes, logistique_hebergement],
        system_prompt=(
            "Tu es logisticien de crise. Pour le lieu indiqué, appelle "
            "accessibilite_routes ET logistique_hebergement. Liste les ouvrages "
            "routiers à risque de coupure (gués, ponts, passages bas) et les sites "
            "d'hébergement avec leur capacité estimée. Si un besoin en couchages "
            "t'est fourni, COMPARE-le à la capacité totale et signale un éventuel "
            "déficit. Privilégie les hébergements accessibles (routes non coupées). "
            "Reste concis."
        ),
    )

    return expert_risque, expert_population, expert_logistique
