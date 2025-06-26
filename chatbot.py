# chatbot.py

import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. Download dos recursos do NLTK (só na primeira vez)
# --------------------------------------------------
nltk.download('punkt')      # Tokenizador de sentenças e palavras
nltk.download('wordnet')    # Dicionário WordNet para lematização
nltk.download('omw-1.4')    # Suporte a dicionários multilíngues

# --------------------------------------------------
# 2. Leitura e tokenização do corpus
# --------------------------------------------------
with open('data.txt', 'r', encoding='utf-8', errors='ignore') as f:
    texto_raw = f.read().lower()               # converte tudo para minúsculas

sentencas = nltk.sent_tokenize(texto_raw)      # divide em frases
palavras = nltk.word_tokenize(texto_raw)       # divide em palavras

# --------------------------------------------------
# 3. Preparação para lematização e remoção de pontuação
# --------------------------------------------------
lemmatizer = nltk.stem.WordNetLemmatizer()
# dicionário para remover todos os sinais de pontuação
remove_pontuacao = dict((ord(p), None) for p in string.punctuation)

def normaliza_texto(texto: str) -> list[str]:
    """
    Converte texto em lowercase, remove pontuação,
    tokeniza e lematiza cada token.
    """
    texto_sem_ponto = texto.lower().translate(remove_pontuacao)
    tokens = nltk.word_tokenize(texto_sem_ponto)
    return [lemmatizer.lemmatize(tok) for tok in tokens]

# --------------------------------------------------
# 4. Funções de saudação
# --------------------------------------------------
saudacoes_entrada = {'oi', 'olá', 'ola', 'opa', 'eae', 'bom', 'boa'}  # palavras-chave
saudacoes_saida = ['Oi!', 'Olá!', 'E aí, tudo bem?', 'Prazer em falar com você!']

def responde_saudacao(frase: str) -> str | None:
    """
    Se a frase do usuário contiver saudação,
    retorna uma resposta aleatória.
    """
    for palavra in frase.split():
        if palavra.lower() in saudacoes_entrada:
            return random.choice(saudacoes_saida)
    return None

# --------------------------------------------------
# 5. Geração de resposta com TF-IDF + cosseno
# --------------------------------------------------
def gera_resposta(usuario_texto: str) -> str:
    """
    Recebe a entrada do usuário e encontra
    a frase mais similar no corpus.
    """
    # adiciona a entrada do usuário ao final da lista de sentenças
    all_sentences = sentencas + [usuario_texto]
    # cria matriz TF-IDF usando nossa função de normalização
    tfidf = TfidfVectorizer(tokenizer=normaliza_texto, stop_words='english')
    tfidf_matriz = tfidf.fit_transform(all_sentences)
    # calcula similaridade entre última entrada e todas as sentenças
    similaridades = cosine_similarity(tfidf_matriz[-1], tfidf_matriz)
    # obtém índice da segunda frase mais semelhante (a última é ela mesma)
    idx_melhor = similaridades.argsort()[0][-2]
    score = similaridades.flatten()[idx_melhor]
    # se não encontrou nada relevante, pede desculpas
    if score == 0:
        return 'Desculpe, não entendi.'
    # caso contrário, retorna a frase do corpus
    return sentencas[idx_melhor]

# --------------------------------------------------
# 6. Laço principal de interação
# --------------------------------------------------
def main():
    print('Olá! Eu sou o Bot de Aprendizagem. Para sair, digite "bye".')
    while True:
        entrada = input('Você: ').strip().lower()
        if entrada == 'bye':
            print('Bot: Até mais!')
            break

        # se for “obrigado” encerra amigavelmente
        if entrada in {'obrigado', 'obrigada', 'valeu'}:
            print('Bot: De nada!')
            continue

        # tentamos responder a uma saudação
        saud = responde_saudacao(entrada)
        if saud:
            print('Bot:', saud)
            continue

        # gera resposta via TF-IDF
        resposta = gera_resposta(entrada)
        print('Bot:', resposta)

if __name__ == '__main__':
    main()
