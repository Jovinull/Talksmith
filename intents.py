import re
import random
from preprocessing import normalize, remove_accents

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

def respond_greeting(text: str) -> str|None:
    if any(w in normalize(text) for w in GREET_IN):
        return random.choice(GREET_OUT)


def find_definition(text: str, sentences: list[str]) -> str|None:
    norm = remove_accents(text.lower())
    m = re.search(r'(?:defina|o\s*que\s*e|oque|me\s*explique)\s+(.+)', norm)
    if not m:
        return None
    term = m.group(1).strip(' ?.')
    results = [s for s in sentences if remove_accents(term) in remove_accents(s)]
    return results[0] if results else None


def list_topics(text: str) -> str|None:
    norm = remove_accents(text.lower())
    if re.search(r'\b(cite|liste|quais)\b.*\b(mat(e|é)rias|disciplinas)\b', norm):
        return ', '.join(AREAS_MATEMATICA[:5])


def inventor_math(text: str) -> str|None:
    norm = remove_accents(text.lower())
    if re.search(r'\bquem\b.*\binventou\b.*\bmatem[aá]tica\b', norm):
        return ("A matemática não foi inventada por uma única pessoa; "
                "ela evoluiu com contribuições de várias civilizações ao longo de milênios.")