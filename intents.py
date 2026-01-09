from __future__ import annotations

import re
import random

from preprocessing import remove_accents
from retrieval import Retriever

GREET_IN = {"oi", "ola", "olá", "opa", "eae", "eai", "bom dia", "boa tarde", "boa noite"}
GREET_OUT = ["Oi!", "Olá!", "E aí, tudo bem?", "Prazer em falar com você!"]

AREAS_MATEMATICA = [
    "teoria dos números",
    "geometria",
    "álgebra",
    "cálculo e análise",
    "matemática discreta",
    "lógica matemática",
    "probabilidade",
    "estatística e decisão",
]


def respond_greeting(text: str) -> str | None:
    norm = remove_accents((text or "").lower()).strip()

    # pega 1-2 palavras iniciais pra não depender de stopwords/lemmas
    start = " ".join(norm.split()[:2])

    if norm in GREET_IN or start in GREET_IN or any(norm.startswith(g) for g in ("oi", "ola", "olá", "opa", "eae", "eai")):
        return random.choice(GREET_OUT)
    return None


def _extract_definition_term(text: str) -> str | None:
    norm = remove_accents((text or "").lower()).strip()

    # padrões comuns
    m = re.search(r"\b(o\s*que\s*e|o\s*que\s*é|oque\s*e|oque\s*é|defina|me\s*explique|explique)\b\s+(.+)", norm)
    if not m:
        return None

    term = (m.group(2) or "").strip(" ?.!\t\r\n")
    if len(term) < 2:
        return None
    return term


def find_definition(text: str, retriever: Retriever) -> str | None:
    term = _extract_definition_term(text)
    if not term:
        return None

    # para "o que é X", buscar por X costuma ser melhor do que buscar a frase inteira
    hits = retriever.top_k_hybrid(term, k=1, threshold=0.14)
    return hits[0] if hits else None


def list_topics(text: str) -> str | None:
    norm = remove_accents((text or "").lower())
    if re.search(r"\b(cite|liste|quais)\b.*\b(areas|mat(e|é)rias|disciplinas|ramos)\b", norm):
        return ", ".join(AREAS_MATEMATICA)
    return None


def inventor_math(text: str) -> str | None:
    norm = remove_accents((text or "").lower())
    if re.search(r"\bquem\b.*\binventou\b.*\bmatem[aá]tica\b", norm):
        return (
            "A matemática não foi inventada por uma única pessoa; ela evoluiu ao longo de milênios "
            "com contribuições de várias civilizações e matemáticos."
        )
    return None
