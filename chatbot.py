import re
import random
import unicodedata
import string

import nltk
from nltk import sent_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# --------------------------------------------------
# 1. Inicialização e downloads
# --------------------------------------------------
nltk.download('punkt', quiet=True)
nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])

# --------------------------------------------------
# 2. Leitura do corpus
# --------------------------------------------------
with open('data.txt', 'r', encoding='utf-8', errors='ignore') as f:
    raw_text = f.read().lower()
sentences = sent_tokenize(raw_text)

# --------------------------------------------------
# 3. Normalização (remoção de acentos + lematização)
# --------------------------------------------------
_remove_punct = str.maketrans('', '', string.punctuation)

def remove_accents(text: str) -> str:
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def normalize(text: str) -> list[str]:
    txt = remove_accents(text.lower())
    doc = nlp(txt)
    return [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]

# --------------------------------------------------
# 4. TF-IDF com uni‐grams e bi‐grams + filtros
# --------------------------------------------------
vectorizer = TfidfVectorizer(
    tokenizer=normalize,
    lowercase=False,
    stop_words=None,
    ngram_range=(1, 2),
    max_df=0.85,
    min_df=2
)
tfidf_matrix = vectorizer.fit_transform(sentences)

# --------------------------------------------------
# 5. BM25 indexação
# --------------------------------------------------
tokenized = [normalize(s) for s in sentences]
bm25 = BM25Okapi(tokenized)

# --------------------------------------------------
# 6. Saudação
# --------------------------------------------------
GREET_IN = {'oi','olá','ola','opa','eae','bom','boa'}
GREET_OUT = ['Oi!','Olá!','E aí, tudo bem?','Prazer em falar com você!']

def respond_greeting(text: str) -> str|None:
    if any(w in normalize(text) for w in GREET_IN):
        return random.choice(GREET_OUT)
    return None

# --------------------------------------------------
# 7. Definição explícita (normalize + regex robusto)
# --------------------------------------------------
def find_definition(text: str) -> str|None:
    """
    Captura variações como:
      - defina X
      - o que é X
      - o que e X
      - oque é X
    """
    norm = remove_accents(text.lower())
    m = re.search(r'(?:defina|o\s*que\s*e|oque)\s+(.+)', norm)
    if not m:
        return None
    term = m.group(1).strip(' ?.')
    for sent in sentences:
        if remove_accents(term) in remove_accents(sent):
            return sent
    return None

# --------------------------------------------------
# 8. Listagem de “matérias” e “quem inventou”
# --------------------------------------------------
AREAS_MATEMATICA = [
    "teoria dos números", "geometria",
    "álgebra", "cálculo e análise",
    "matemática discreta", "lógica matemática",
    "probabilidade", "estatística e decisão"
]

def list_topics(text: str) -> str|None:
    norm = remove_accents(text.lower())
    if re.search(r'\b(cite|liste|quais)\b.*\b(mat(e|é)rias|disciplinas)\b', norm):
        return ', '.join(AREAS_MATEMATICA[:5])  # você pode ajustar quantos quiser
    return None

def inventor_math(text: str) -> str|None:
    norm = remove_accents(text.lower())
    if re.search(r'\bquem\b.*\binventou\b.*\bmatem[aá]tica\b', norm):
        return ("A matemática não foi inventada por uma única pessoa; "
                "ela evoluiu com contribuições de várias civilizações ao longo de milênios.")
    return None

# --------------------------------------------------
# 9. Top-K + Threshold (TF-IDF)
# --------------------------------------------------
SIM_THRESHOLD = 0.10

def top_k_tfidf(user_text: str, k: int = 3) -> list[str]:
    v = vectorizer.transform([user_text])
    sims = cosine_similarity(v, tfidf_matrix).flatten()
    idxs = sims.argsort()[-k:][::-1]
    return [sentences[i] for i in idxs if sims[i] >= SIM_THRESHOLD]

# --------------------------------------------------
# 10. Top-K BM25 de fallback
# --------------------------------------------------
def top_k_bm25(user_text: str, k: int = 3) -> list[str]:
    tokens = normalize(user_text)
    scores = bm25.get_scores(tokens)
    idxs = scores.argsort()[-k:][::-1]
    return [sentences[i] for i in idxs if scores[i] > 0]

# --------------------------------------------------
# 11. Geração de resposta
# --------------------------------------------------
def generate_response(user_text: str) -> str:
    # 1) Saudação
    if (g := respond_greeting(user_text)):
        return g

    # 2) Pergunta de definição
    if (d := find_definition(user_text)):
        return d

    # 3) Listagem de matérias
    if (lst := list_topics(user_text)):
        return lst

    # 4) Quem inventou?
    if (inv := inventor_math(user_text)):
        return inv

    # 5) TF-IDF top-k
    tfidf_ans = top_k_tfidf(user_text)
    if tfidf_ans:
        return '\n'.join(tfidf_ans)

    # 6) BM25 fallback
    bm25_ans = top_k_bm25(user_text)
    if bm25_ans:
        return '\n'.join(bm25_ans)

    # 7) Sem resposta
    return 'Desculpe, não encontrei uma resposta clara para isso.'

# --------------------------------------------------
# 12. Loop principal
# --------------------------------------------------
def main():
    print('Olá! Eu sou o Bot de Aprendizagem. Para sair, digite "bye".')
    while True:
        user = input('Você: ').strip()
        if user.lower() == 'bye':
            print('Bot: Até mais!')
            break
        if user.lower() in {'obrigado','obrigada','valeu'}:
            print('Bot: De nada!')
            continue

        print('Bot:', generate_response(user))

if __name__ == '__main__':
    main()
