from __future__ import annotations

from datetime import datetime
import json
import random

from retrieval import Retriever
import intents


class BotEngine:
    def __init__(
        self,
        retriever: Retriever,
        k: int = 3,
        threshold: float = 0.18,
        history_path: str = "history.json",
    ):
        self.retriever = retriever
        self.k = k
        self.threshold = threshold
        self.history: list[dict] = []
        self.history_path = history_path

        self.default_responses = [
            "Hmm... não encontrei um trecho bom no meu material para isso. Pode perguntar de outro jeito?",
            "Não consegui conectar sua pergunta com o conteúdo que eu tenho aqui. Pode reformular?",
            "Essa eu não achei no texto. Tente ser mais específico no termo/assunto 😊",
        ]

    def get_response(self, user_input: str) -> str:
        ui = (user_input or "").strip()
        if not ui:
            return random.choice(self.default_responses)

        if ui.lower() in {"obrigado", "obrigada", "valeu"}:
            return "De nada!"

        # Intents “rápidos”
        resp = (
            intents.respond_greeting(ui)
            or intents.inventor_math(ui)
            or intents.list_topics(ui)
            or intents.find_definition(ui, self.retriever)
        )

        # Retrieval híbrido (com histórico simples)
        if not resp:
            query = self._build_query(ui)
            hits = self.retriever.top_k_hybrid(query, k=self.k, threshold=self.threshold)

            if hits:
                resp = self._compose_answer(hits)
            else:
                resp = random.choice(self.default_responses)

        self._save_to_history(ui, resp)
        return resp

    def _build_query(self, ui: str) -> str:
        if not self.history:
            return ui

        last_user = self.history[-1].get("user", "")
        # contexto leve: última pergunta + atual
        return f"{last_user} {ui}".strip()

    def _compose_answer(self, hits: list[str]) -> str:
        # resposta mais “inteira”: 1 trecho principal + 1 complemento (se existir)
        if not hits:
            return random.choice(self.default_responses)

        main = hits[0].strip()

        if len(hits) >= 2:
            extra = hits[1].strip()
            # evita repetir se for muito parecido
            if extra and extra != main:
                return f"{main}\n\n{extra}"

        return main

    def _save_to_history(self, user_text: str, bot_text: str):
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user": user_text,
                "bot": bot_text,
            }
        )

    def save_history_to_file(self):
        if self.history:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
