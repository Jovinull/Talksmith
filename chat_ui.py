from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext
from pathlib import Path

from corpus import load_chunks_from_file
from retrieval import Retriever
from bot_engine import BotEngine


class ChatUI:
    def __init__(
        self,
        corpus_path: str,
        k: int = 3,
        threshold: float = 0.18,
        chunk_size: int = 5,
        overlap: int = 2,
        index_path: str = "index.joblib",
    ):
        self.corpus_path = corpus_path
        self.index_path = index_path

        # carrega índice pronto se existir; se não, constrói e salva
        if Path(index_path).exists():
            self.retriever = Retriever.load(index_path)
        else:
            docs = load_chunks_from_file(corpus_path, chunk_size=chunk_size, overlap=overlap)
            self.retriever = Retriever(docs)
            self.retriever.save(index_path)

        self.bot = BotEngine(self.retriever, k=k, threshold=threshold)

        # UI
        self.bg_color = "#2e2e2e"
        self.chat_bg = "#1e1e1e"
        self.input_bg = "#3e3e3e"
        self.bot_text_color = "#f0f0f0"
        self.user_text_color = "#00ff99"
        self.button_color = "#00b386"
        self.font = ("Segoe UI", 12)

        self.window = tk.Tk()
        self.window.title("ChatBot Educacional (RAG simples)")
        self.window.geometry("780x620")
        self.window.configure(bg=self.bg_color)

        self.chat_area = scrolledtext.ScrolledText(
            self.window,
            wrap=tk.WORD,
            font=self.font,
            bg=self.chat_bg,
            fg=self.bot_text_color,
            insertbackground="white",
            borderwidth=0,
            relief="flat",
        )
        self.chat_area.pack(padx=12, pady=12, fill=tk.BOTH, expand=True)
        self.chat_area.configure(state="disabled")

        self.entry_frame = tk.Frame(self.window, bg=self.bg_color)
        self.entry_frame.pack(fill=tk.X, padx=10, pady=10)

        self.entry = tk.Entry(
            self.entry_frame,
            font=self.font,
            bg=self.input_bg,
            fg="white",
            insertbackground="white",
            relief="flat",
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(
            self.entry_frame,
            text="Enviar",
            font=self.font,
            bg=self.button_color,
            fg="white",
            activebackground="#009970",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=6,
            command=self.send_message,
        )
        self.send_button.pack(side=tk.RIGHT)

        # tags configuradas 1x
        self.chat_area.tag_config("Você", foreground=self.user_text_color)
        self.chat_area.tag_config("Bot", foreground=self.bot_text_color)

        self.display_message("Bot", "Olá! Pergunte algo sobre o conteúdo do seu data.txt. Para sair, digite 'bye'.")

    def display_message(self, sender: str, message: str):
        self.chat_area.configure(state="normal")
        self.chat_area.insert(tk.END, f"{sender}: {message}\n\n", sender)
        self.chat_area.configure(state="disabled")
        self.chat_area.yview(tk.END)

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return

        self.entry.delete(0, tk.END)
        self.display_message("Você", user_input)

        if user_input.lower() == "bye":
            self.display_message("Bot", "Até mais!")
            self.bot.save_history_to_file()
            self.window.quit()
            return

        response = self.bot.get_response(user_input)
        self.display_message("Bot", response)

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    ChatUI("data.txt").run()
