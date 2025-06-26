import nltk
import string
import random
import unicodedata

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. Download dos recursos do NLTK (só na primeira vez)
# --------------------------------------------------
nltk.download('punkt')        # Tokenizador de sentenças e palavras
nltk.download('stopwords')    # Lista de stop-words em vários idiomas

# --------------------------------------------------
# 2. Leitura e tokenização do corpus
# --------------------------------------------------
with open('data.txt', 'r', encoding='utf-8', errors='ignore') as f:
    texto_raw = f.read().lower()

# sentenças originais para resposta
sentencas = nltk.sent_tokenize(texto_raw)

# --------------------------------------------------
# 3. Preparação de normalização (acentos, pontuação, stop-words, stemmer)
# --------------------------------------------------
# remover acentuação
def remove_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize('NFKD', texto)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])

# dicionário para remover pontuação
remove_pontuacao = str.maketrans('', '', string.punctuation)

# stemmer para português
stemmer_pt = SnowballStemmer('portuguese')

# lista de stop-words em português
stopwords_pt = set(stopwords.words('portuguese'))

def normaliza_texto(texto: str) -> list[str]:
    """
    Remove acentos e pontuação, tokeniza, filtra stop-words
    e aplica stemming em cada token.
    """
    # retira acentos e deixa em lowercase
    texto = remove_acentos(texto.lower())
    # retira pontuação
    texto = texto.translate(remove_pontuacao)
    # tokeniza
    tokens = nltk.word_tokenize(texto)
    # filtra alfanuméricos e stop-words, aplica stemmer
    resultado = []
    for tok in tokens:
        if tok.isalpha() and tok not in stopwords_pt:
            resultado.append(stemmer_pt.stem(tok))
    return resultado

# --------------------------------------------------
# 4. Funções de saudação
# --------------------------------------------------
saudacoes_entrada = {'oi', 'olá', 'ola', 'opa', 'eae', 'bom', 'boa'}
saudacoes_saida = ['Oi!', 'Olá!', 'E aí, tudo bem?', 'Prazer em falar com você!']

def responde_saudacao(frase: str) -> str | None:
    """
    Se a saudação estiver entre os tokens normalizados,
    retorna uma resposta aleatória.
    """
    tokens = normaliza_texto(frase)
    for saud in saudacoes_entrada:
        if saud in tokens:
            return random.choice(saudacoes_saida)
    return None

# --------------------------------------------------
# 5. Pré-computar TF-IDF (só uma vez)
# --------------------------------------------------
vectorizer = TfidfVectorizer(
    tokenizer=normaliza_texto,
    lowercase=False,            # já normalizamos antes
    stop_words=None             # já removemos stop-words manualmente
)
tfidf_matrix = vectorizer.fit_transform(sentencas)

def gera_resposta(usuario_texto: str) -> str:
    """
    Transforma a entrada do usuário e compara por similaridade
    contra o corpus pré-processado.
    """
    # transforma apenas a nova frase
    v = vectorizer.transform([usuario_texto])
    # calcula similaridade contra todas as sentenças
    sims = cosine_similarity(v, tfidf_matrix).flatten()
    idx = sims.argmax()
    if sims[idx] == 0:
        return 'Desculpe, não entendi.'
    return sentencas[idx]

# --------------------------------------------------
# 6. Laço principal de interação
# --------------------------------------------------
def main():
    print('Olá! Eu sou o Bot de Aprendizagem. Para sair, digite "bye".')
    while True:
        entrada_raw = input('Você: ')
        entrada = entrada_raw.strip()
        if entrada.lower() == 'bye':
            print('Bot: Até mais!')
            break

        if entrada.lower() in {'obrigado', 'obrigada', 'valeu'}:
            print('Bot: De nada!')
            continue

        # resposta de saudação
        saud = responde_saudacao(entrada)
        if saud:
            print('Bot:', saud)
            continue

        # resposta via TF-IDF + cosseno
        resposta = gera_resposta(entrada)
        print('Bot:', resposta)

if __name__ == '__main__':
    main()
