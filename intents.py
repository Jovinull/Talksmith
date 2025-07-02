import re
import random
import spacy
from spacy.matcher import Matcher
from preprocessing import normalize, remove_accents

# spaCy e Matcher para “me diga o que é X”
_nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner', 'tagger'])
_matcher = Matcher(_nlp.vocab)
_pattern = [
    {"LOWER": "me"},
    {"LEMMA": "dizer"},
    {"LOWER": "o"},
    {"LOWER": "que"},
    {"LOWER": "é"},
    {"OP": "*"}
]
_matcher.add("EXPLAIN", [ _pattern ])


# Saudações
GREET_IN = {'oi', 'olá', 'ola', 'opa', 'eae', 'bom', 'boa'}
GREET_OUT = ['Oi!', 'Olá!', 'E aí, tudo bem?', 'Prazer em falar com você!']

# Lista de áreas de Matemática
AREAS_MATEMATICA = [
    "teoria dos números", "geometria",
    "álgebra", "cálculo e análise",
    "matemática discreta", "lógica matemática",
    "probabilidade", "estatística e decisão"
]

def respond_greeting(text: str) -> str | None:
    if any(w in normalize(text) for w in GREET_IN):
        return random.choice(GREET_OUT)

def find_definition(text: str, sentences: list[str]) -> str | None:
    norm = remove_accents(text.lower())
    # 1) tenta spaCy Matcher
    doc = _nlp(norm)
    matches = _matcher(doc)
    if matches:
        _, start, end = matches[0]
        term = doc[end:].text.strip(' ?.')
    else:
        # 2) fallback regex
        m = re.search(r'(?:defina|o\s*que\s*e|oque|me\s*explique)\s+(.+)', norm)
        if not m:
            return None
        term = m.group(1).strip(' ?.')
    # busca na lista de sentenças
    term_norm = remove_accents(term)
    for s in sentences:
        if term_norm in remove_accents(s.lower()):
            return s
    return None

def list_topics(text: str) -> str | None:
    norm = remove_accents(text.lower())
    if re.search(r'\b(cite|liste|quais)\b.*\b(mat(e|é)rias|disciplinas)\b', norm):
        return ', '.join(AREAS_MATEMATICA[:5])

def inventor_math(text: str) -> str | None:
    norm = remove_accents(text.lower())
    if re.search(r'\bquem\b.*\binventou\b.*\bmatem[aá]tica\b', norm):
        return ("A matemática não foi inventada por uma única pessoa; "
                "ela evoluiu com contribuições de várias civilizações ao longo de milênios.")
