from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import Iterable

import spacy

_NLP = None


def get_nlp():
    """
    Carrega spaCy UMA vez e reaproveita.
    Importante: não desabilitar tagger/morph, porque isso afeta a lematização.
    """
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
        except OSError as e:
            raise RuntimeError(
                "Modelo spaCy 'pt_core_news_sm' não encontrado. "
                "Instale com: python -m spacy download pt_core_news_sm"
            ) from e
    return _NLP


def remove_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@lru_cache(maxsize=4096)
def normalize(text: str) -> tuple[str, ...]:
    """
    Normaliza um texto em tokens lematizados (tupla para cache).
    - lower
    - remove acentos
    - remove stopwords
    - mantém apenas palavras (is_alpha)
    """
    txt = remove_accents(text).lower()
    nlp = get_nlp()
    doc = nlp(txt)

    out: list[str] = []
    for tok in doc:
        if not tok.is_alpha:
            continue
        if tok.is_stop:
            continue
        lemma = (tok.lemma_ or "").strip()
        if lemma:
            out.append(lemma)
    return tuple(out)


def normalize_join(text: str) -> str:
    return " ".join(normalize(text))


def normalize_many(texts: Iterable[str], batch_size: int = 64) -> list[list[str]]:
    """
    Normaliza vários textos usando nlp.pipe (bem mais rápido no corpus).
    """
    nlp = get_nlp()
    cleaned = (remove_accents(t).lower() for t in texts)

    results: list[list[str]] = []
    for doc in nlp.pipe(cleaned, batch_size=batch_size):
        toks: list[str] = []
        for tok in doc:
            if tok.is_alpha and (not tok.is_stop):
                lemma = (tok.lemma_ or "").strip()
                if lemma:
                    toks.append(lemma)
        results.append(toks)

    return results
