# TalkSmith

Um chatbot de Aprendizado de Máquina que responde a perguntas em português usando técnicas de TF-IDF e BM25 sobre um corpus de texto, agora com uma interface gráfica simples feita em `tkinter`.

---

## Descrição

Este projeto implementa um assistente de conversação simples em Python, capaz de:

- Cumprimentar o usuário de forma natural.  
- Capturar definições de termos (e.g. “o que é X?”, “defina X”).  
- Listar tópicos de matemática e contextualizar perguntas como “quais matérias?”.  
- Informar sobre quem “inventou” a matemática.  
- Buscar respostas relevantes no corpus usando **TF-IDF** (uni-grams e bi-grams) e, em caso de baixa similaridade, recorrer ao **BM25**.  
- Normalizar texto (remoção de acentos, lematização e filtros de stopwords).
- Interagir por meio de uma **interface gráfica amigável**, com design escuro, botões estilizados e rolagem automática da conversa.

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
   git clone https://github.com/seu-usuario/talksmith.git
   cd talksmith
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

## Estrutura do Projeto

```text
talksmith/
├── chat_ui.py             # Interface gráfica com tkinter
├── data.txt               # Corpus de conhecimento em texto puro
├── intents.py             # Regras manuais de intenções (saudação, definição etc)
├── preprocessing.py       # Lematização, remoção de acentos e stopwords
├── retrieval.py           # Implementação do TF-IDF, BM25 e busca
├── requirements.txt       # Dependências do projeto
├── README.md              # Documentação geral
├── .env                   # Variáveis de ambiente (local)
├── .env.example           # Modelo base de .env
├── .gitignore             # Arquivos ignorados no Git
└── venv/                  # Ambiente virtual (local)
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
   * **BM25** como fallback para consultas menos comuns.

4. **Geração de Resposta**

   * Verifica intenções manuais (saudações, definição, tópicos etc.).
   * Se necessário, usa recuperação de similaridade textual (TF-IDF → BM25).
   * Caso não encontre resposta com confiança, exibe uma mensagem padrão.

5. **Interface Gráfica com `tkinter`**

   * Interface escura e responsiva.
   * Campo de entrada + botão de envio.
   * Área de chat com rolagem e destaque visual entre usuário e bot.

---

## Uso

Execute a interface gráfica:

```bash
python chat_ui.py
```

* Digite perguntas diretamente no campo inferior.
* Pressione Enter ou clique em “Enviar”.
* Para sair, digite `bye`.

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
