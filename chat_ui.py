import tkinter as tk
from tkinter import scrolledtext
from nltk import sent_tokenize

from retrieval import Retriever
import intents


def load_corpus(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read().lower()
    return sent_tokenize(raw)


class ChatUI:
    def __init__(self, corpus_path: str, k: int = 3, threshold: float = 0.1):
        self.sentences = load_corpus(corpus_path)
        self.retriever = Retriever(self.sentences)
        self.k = k
        self.threshold = threshold

        # Cores e fontes
        self.bg_color = "#2e2e2e"
        self.chat_bg = "#1e1e1e"
        self.input_bg = "#3e3e3e"
        self.bot_text_color = "#f0f0f0"
        self.user_text_color = "#00ff99"
        self.button_color = "#00b386"
        self.font = ("Segoe UI", 12)

        self.window = tk.Tk()
        self.window.title("ChatBot Educacional")
        self.window.geometry("700x600")
        self.window.configure(bg=self.bg_color)

        self.chat_area = scrolledtext.ScrolledText(self.window, 
            wrap=tk.WORD,
            font=self.font,
            bg=self.chat_bg,
            fg=self.bot_text_color,
            insertbackground="white",
            borderwidth=0,
            relief="flat"
        )
        self.chat_area.pack(padx=12, pady=12, fill=tk.BOTH, expand=True)
        self.chat_area.configure(state='disabled')

        self.entry_frame = tk.Frame(self.window, bg=self.bg_color)
        self.entry_frame.pack(fill=tk.X, padx=10, pady=10)

        self.entry = tk.Entry(self.entry_frame, font=self.font, bg=self.input_bg, fg="white", insertbackground="white", relief="flat")
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.entry_frame, text="Enviar", font=self.font,
                                     bg=self.button_color, fg="white",
                                     activebackground="#009970", activeforeground="white",
                                     relief="flat", padx=12, pady=6, command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

        self.display_message("Bot", "Olá! Eu sou o Bot de Aprendizagem. Para sair, digite 'bye'.")

    def display_message(self, sender: str, message: str):
        self.chat_area.configure(state='normal')
        prefix = "Você" if sender.lower() == "você" else "Bot"
        color = self.user_text_color if sender.lower() == "você" else self.bot_text_color
        self.chat_area.insert(tk.END, f"{prefix}: {message}\n", prefix)
        self.chat_area.tag_config(prefix, foreground=color)
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.entry.delete(0, tk.END)
        self.display_message("Você", user_input)

        if user_input.lower() == "bye":
            self.display_message("Bot", "Até mais!")
            self.window.quit()
            return

        response = self.get_response(user_input)
        self.display_message("Bot", response)

    def get_response(self, user: str) -> str:
        if user.lower() in {'obrigado', 'obrigada', 'valeu'}:
            return "De nada!"

        resp = (intents.respond_greeting(user) or
                intents.find_definition(user, self.sentences) or
                intents.list_topics(user) or
                intents.inventor_math(user))

        if not resp:
            tfidf_ans = self.retriever.top_k_tfidf(user, k=self.k, threshold=self.threshold)
            resp = '\n'.join(tfidf_ans) if tfidf_ans else None
        if not resp:
            bm25_ans = self.retriever.top_k_bm25(user, k=self.k)
            resp = '\n'.join(bm25_ans) if bm25_ans else None
        if not resp:
            resp = "Desculpe, não encontrei uma resposta clara para isso."

        return resp

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    chat = ChatUI("data.txt")
    chat.run()
