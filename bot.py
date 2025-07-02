import argparse
import logging
from nltk import sent_tokenize
from preprocessing import normalize, remove_accents
from retrieval import Retriever
import intents

# Configuração de logging

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO
    )

# Carrega e tokeniza o corpus

def load_corpus(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read().lower()
    return sent_tokenize(raw)


def main():
    parser = argparse.ArgumentParser(description='Bot de Aprendizagem')
    parser.add_argument('corpus', help='Caminho para o arquivo de texto')
    parser.add_argument('-k', type=int, default=3, help='Número de respostas')
    parser.add_argument('--threshold', type=float, default=0.1)
    args = parser.parse_args()

    setup_logging()
    sentences = load_corpus(args.corpus)
    retriever = Retriever(sentences)
    logging.info('Índices carregados. Bot pronto!')

    print('Olá! Eu sou o Bot de Aprendizagem. Para sair, digite "bye".')
    while True:
        user = input('Você: ').strip()
        if user.lower() == 'bye':
            print('Bot: Até mais!')
            break
        if user.lower() in {'obrigado', 'obrigada', 'valeu'}:
            print('Bot: De nada!')
            continue

        # Tenta reconhecer intent
        resp = (intents.respond_greeting(user) or
                intents.find_definition(user, sentences) or
                intents.list_topics(user) or
                intents.inventor_math(user))

        # Retrieval híbrido TF-IDF -> BM25
        if not resp:
            tfidf_ans = retriever.top_k_tfidf(user, k=args.k, threshold=args.threshold)
            resp = '\n'.join(tfidf_ans) if tfidf_ans else None
        if not resp:
            bm25_ans = retriever.top_k_bm25(user, k=args.k)
            resp = '\n'.join(bm25_ans) if bm25_ans else None
        if not resp:
            resp = 'Desculpe, não encontrei uma resposta clara para isso.'

        print('Bot:', resp)

if __name__ == '__main__':
    main()
