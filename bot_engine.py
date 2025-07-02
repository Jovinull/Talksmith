from datetime import datetime
import json
import random


class BotEngine:
    def __init__(self, retriever, sentences, k=3, threshold=0.1, history_path="history.json"):
        self.retriever = retriever
        self.sentences = sentences
        self.k = k
        self.threshold = threshold
        self.history = []
        self.history_path = history_path

        self.default_responses = [
            "Hmm... não tenho uma resposta clara para isso agora.",
            "Desculpe, não consegui entender muito bem. Pode reformular?",
            "Essa passou batido por mim! Tente perguntar de outro jeito 😊",
            "Ainda não aprendi sobre isso, mas estou sempre tentando melhorar!"
        ]

    def get_response(self, user_input, intents):
        if user_input.lower() in {'obrigado', 'obrigada', 'valeu'}:
            return "De nada!"

        resp = (intents.respond_greeting(user_input) or
                intents.find_definition(user_input, self.sentences) or
                intents.list_topics(user_input) or
                intents.inventor_math(user_input))

        if not resp:
            tfidf_ans = self.retriever.top_k_tfidf(user_input, k=self.k, threshold=self.threshold)
            resp = '\n'.join(tfidf_ans) if tfidf_ans else None
        if not resp:
            bm25_ans = self.retriever.top_k_bm25(user_input, k=self.k)
            resp = '\n'.join(bm25_ans) if bm25_ans else None
        if not resp:
            resp = random.choice(self.default_responses)

        self._save_to_history(user_input, resp)
        return resp

    def _save_to_history(self, user_text, bot_text):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_text,
            "bot": bot_text
        })

    def save_history_to_file(self):
        if self.history:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
