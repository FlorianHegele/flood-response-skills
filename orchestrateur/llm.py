# -*- coding: utf-8 -*-
"""Sélection centralisée du backend LLM (cf. cours, chap. 1/5).

Bascule Mistral (cloud, clé gratuite) ↔ Ollama (local, sans clé) ↔ Anthropic
(Claude, cloud) via la variable d'environnement ``LLM_BACKEND``. Centraliser ce
choix ici évite de le répéter dans chaque agent et simplifie l'évolution du projet.
"""

import os


def get_llm(temperature: float = 0):
    """Renvoie un chat model LangChain selon ``LLM_BACKEND`` (défaut : mistral).

    - ``mistral``   : ChatMistralAI (``mistral-small-latest``), nécessite ``MISTRAL_API_KEY``.
    - ``ollama``    : ChatOllama (modèle ``OLLAMA_MODEL``, défaut ``qwen2.5:3b``).
    - ``anthropic`` : ChatAnthropic (modèle ``ANTHROPIC_MODEL``, défaut ``claude-opus-4-8``),
      nécessite ``ANTHROPIC_API_KEY``.
    """
    backend = os.getenv("LLM_BACKEND", "mistral").strip().lower()
    if backend == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model="mistral-small-latest", temperature=temperature)
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"), temperature=temperature
        )
    if backend in ("anthropic", "claude"):
        from langchain_anthropic import ChatAnthropic

        # NB : Opus 4.8/4.7 rejettent `temperature` (erreur 400). On ne le passe
        # donc pas pour ce backend ; le défaut du modèle s'applique.
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"),
        )
    raise ValueError(
        f"LLM_BACKEND inconnu : {backend!r} (attendu 'mistral', 'ollama' ou 'anthropic')"
    )
