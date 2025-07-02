# TalkSmith

Um chatbot de Aprendizado de Máquina que responde a perguntas em português usando técnicas de TF-IDF e BM25 sobre um corpus de texto.

---

## Descrição

Este projeto implementa um assistente de conversação simples em Python, capaz de:

- Cumprimentar o usuário de forma natural.  
- Capturar definições de termos (e.g. “o que é X?”, “defina X”).  
- Listar tópicos de matemática e contextualizar perguntas como “quais matérias?”.  
- Informar sobre quem “inventou” a matemática.  
- Buscar respostas relevantes no corpus usando **TF-IDF** (uni-grams e bi-grams) e, em caso de baixa similaridade, recorrer ao **BM25**.  
- Normalizar texto (remoção de acentos, lematização e filtros de stopwords).

---

## Tecnologias e Dependências

- **Python** ≥ 3.8  
- [nltk](https://pypi.org/project/nltk) (3.9.1)  
- [spacy](https://pypi.org/project/spacy) (+ modelo `pt_core_news_sm`)  
- [scikit-learn](https://pypi.org/project/scikit-learn) (1.7.0)  
- [rank_bm25](https://pypi.org/project/rank-bm25)  
- Outros: `click`, `colorama`, `joblib`, `numpy`, `regex`, `scipy`, `threadpoolctl`, `tqdm`

> As versões exatas estão listadas no arquivo `requirements.txt`.

---

## Instalação

1. **Clone** o repositório  
   ```bash
   git clone https://github.com/seu-usuario/chat-ai-bot.git
   cd chat-ai-bot
   ```

2. **Crie e ative** um ambiente virtual

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Instale** as dependências

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Baixe** o modelo de português do spaCy

   ```bash
   python -m spacy download pt_core_news_sm
   ```

---

## Configuração de Ambiente

Todos os parâmetros sensíveis (e.g. conexão com banco de dados) devem ficar em variáveis de ambiente. Copie o arquivo de exemplo e preencha os valores reais:

```bash
cp .env.example .env
```

`.env.example`:

```dotenv
# URL de conexão com o PostgreSQL
DATABASE_URL=postgresql://<usuario>:<senha>@<host>:<porta>/<database>
```

> A variável `DATABASE_URL` é utilizada para gravação de logs, histórico de conversas ou análises avançadas (se implementadas no futuro).

---

## Estrutura do Projeto

```
chat-ai-bot/
├── data.txt                # Corpus de texto (frases para consulta)
├── requirements.txt        # Lista completa de dependências
├── .env.example            # Exemplo de variáveis de ambiente
├── bot.py                  # Script principal do chatbot
└── utils/
    ├── preprocess.py       # Funções de normalização e lematização
    └── retrieval.py        # Implementação de TF-IDF, BM25 e geração de resposta
```

---

## Como Funciona

1. **Leitura e Tokenização**

   * Carrega `data.txt` e separa em sentenças usando `nltk.sent_tokenize`.

2. **Normalização**

   * Remove acentos e pontuação.
   * Lematiza e filtra stopwords usando spaCy.

3. **Índices de Pesquisa**

   * **TF-IDF** com n-grams (uni-grams e bi-grams), filtrando termos muito frequentes ou raros.
   * **BM25** como fallback para garantirmos respostas mesmo em consultas não cobertas pelo TF-IDF.

4. **Geração de Resposta**

   * Verifica saudações, pedidos de definição, listagem de matérias e perguntas sobre quem “inventou” a matemática.
   * Em seguida, realiza busca nos índices (TF-IDF → BM25).
   * Caso nada seja encontrado acima do limiar, retorna mensagem padrão de “sem resposta clara”.

5. **Loop Interativo**

   * Exibe prompt no terminal, aguarda entrada do usuário e responde até que seja digitado “bye”.

---

## Uso

Execute o bot:

```bash
python bot.py
```

* Digite perguntas em português.
* Para sair, digite `bye`.
* Para agradecimentos, o bot responde automaticamente.

---

## Contribuição

1. Faça um **fork** do repositório.
2. Crie uma **branch** (`git checkout -b feature/nova-funcionalidade`).
3. Faça **commit** das suas alterações (`git commit -m "Adiciona X"`).
4. **Push** para sua branch (`git push origin feature/nova-funcionalidade`).
5. Abra um **Pull Request** e aguarde a revisão.

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE). Sinta-se à vontade para usar, modificar e distribuir.
