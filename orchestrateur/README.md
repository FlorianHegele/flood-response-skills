# Orchestrateur crue — agent + sous-agents experts

Couche d'orchestration au-dessus des 5 skills `flood-response-skills`. À partir
d'une commune en crue, elle produit une **note de décision** complète : danger →
qui évacuer → routes à surveiller → où héberger.

## Architecture

Un **orchestrateur LangGraph** (`StateGraph`) appelle **3 sous-agents experts**
(`create_agent`), chacun outillé de skills regroupés par thème :

```
START → noeud_risque ──[routeur]── vigilance verte/inconnue ─→ synthèse → END
                        │ jaune/orange/rouge
                        └→ population → logistique → synthèse → END

expert_risque      → alerte_crue
expert_population  → demographie_iris + vulnerabilite_bpe
expert_logistique  → accessibilite_routes + logistique_hebergement
```

Pourquoi ce découpage : les skills sont **couplés**. La vigilance *gate* la suite
(on ne mobilise pas si pas de risque), le nombre d'évacués *dimensionne*
l'hébergement (transporté dans l'état partagé), et la synthèse finale agrège le
tout. Le flux étant connu et conditionnel, on l'**encode** (LangGraph) au lieu de
le confier à un superviseur LLM libre. Les sous-agents regroupent ≥ 2 skills pour
raisonner dessus (sinon ils ne seraient que des passe-plats).

Chaque skill est exposé comme un **outil LangChain** (`skills_tools.py`) qui
shelle vers `skills/<skill>/main.py` et relaie son JSON.

## Installation

Avec **uv** (recommandé, cf. cours) :

```bash
cd orchestrateur
uv sync                 # installe langchain, langgraph, le backend, python-dotenv
cp .env.example .env    # puis éditer .env
```

Sans uv, via **pip** dans un venv dédié :

```bash
cd orchestrateur
python3 -m venv .venv && . .venv/bin/activate
pip install "langchain>=1.0" langchain-anthropic langchain-mistralai langchain-ollama "langgraph>=0.2" python-dotenv
cp .env.example .env    # puis éditer .env
```

Configurer `.env` :
- **Anthropic / Claude** (cloud, clé sur https://console.anthropic.com/) :
  `LLM_BACKEND=anthropic` + `ANTHROPIC_API_KEY=sk-ant-...`. Modèle via
  `ANTHROPIC_MODEL` (défaut `claude-opus-4-8` ; `claude-sonnet-4-6` ou
  `claude-haiku-4-5` pour réduire le coût sur un agent à nombreux appels).
- **Mistral** (cloud, clé gratuite sur https://console.mistral.ai/) :
  `LLM_BACKEND=mistral` + `MISTRAL_API_KEY=...`
- **Ollama** (local, sans clé) : `LLM_BACKEND=ollama`, après `ollama pull qwen2.5:3b`.

> Les skills tournent dans le `.venv/` de la racine du dépôt (auto-créé à leur
> premier appel) ; cet environnement-ci ne contient que les dépendances de l'agent.

## Lancement

```bash
uv run python main.py --commune "Alès"          # note de décision
uv run python main.py --commune 30007 --json    # état complet du graphe (debug)
uv run python main.py --commune "Nîmes" --backend ollama
```

> Sans `uv`, remplacer `uv run python` par `.venv/bin/python`.

## Tester la chaîne complète hors période de crue

En vigilance verte, le routeur court-circuite vers la synthèse : population,
logistique et hébergement ne sont pas mobilisés. Pour exercer **toute** la chaîne
sans attendre une vraie crue, `--simuler-vigilance {vert,jaune,orange,rouge}`
superpose un scénario d'aléa de cette couleur sur la **vraie** sortie
d'`alerte-crue` (commune, coordonnées et stations réelles conservées). La
simulation est explicitement étiquetée (`_simulation`) ; tout le pipeline tourne
normalement (l'expert risque résume, le classifieur classe, le gate s'ouvre),
seules les valeurs d'aléa sont injectées.

```bash
uv run python main.py --commune "Alès" --simuler-vigilance rouge          # chaîne complète
uv run python main.py --commune "Alès" --simuler-vigilance rouge --json   # état détaillé
uv run python main.py --commune "Alès" --simuler-vigilance vert           # branche veille
```

> Drapeau de **test uniquement** : il ne reflète aucune donnée Vigicrues/Hub'Eau/
> AROME réelle. À ne jamais utiliser pour une décision opérationnelle.

## Vérifier le graphe

```bash
uv run python -c "from orchestrateur import build_graph; print(build_graph().get_graph().draw_ascii())"
```
