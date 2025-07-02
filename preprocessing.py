import unicodedata
from functools import lru_cache
import spacy

# Pipeline spaCy inicializado sob demanda (lazy loading)
_nlp = None

def _load_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner', 'tagger'])
    return _nlp

# Remove acentos de um texto
def remove_accents(text: str) -> str:
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

# Normalização com lematização e cache
@lru_cache(maxsize=1024)
def normalize(text: str) -> list[str]:
    txt = remove_accents(text.lower())
    nlp = _load_nlp()
    doc = nlp(txt)
    return [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]
