from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import normalize, normalize_join, normalize_many


class Retriever:
    """
    Index híbrido:
    - TF-IDF em cima do texto normalizado (lemmas)
    - BM25 em cima de tokens normalizados
    """

    def __init__(
        self,
        docs: list[str],
        ngram_range: tuple[int, int] = (1, 2),
        max_df: float = 0.9,
        min_df: int = 1,
    ):
        self.docs = docs

        # Normaliza corpus uma única vez (melhora muito o desempenho)
        self.bm25_tokens: list[list[str]] = normalize_many(docs)
        norm_docs = [" ".join(toks) for toks in self.bm25_tokens]

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(norm_docs)
        self.bm25 = BM25Okapi(self.bm25_tokens)

    def save(self, path: str):
        payload = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "docs": self.docs,
            "bm25_tokens": self.bm25_tokens,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> "Retriever":
        payload = joblib.load(path)

        ret = Retriever.__new__(Retriever)
        ret.vectorizer = payload["vectorizer"]
        ret.tfidf_matrix = payload["tfidf_matrix"]
        ret.docs = payload["docs"]
        ret.bm25_tokens = payload["bm25_tokens"]
        ret.bm25 = BM25Okapi(ret.bm25_tokens)
        return ret

    def top_k_tfidf(self, query: str, k: int = 3, threshold: float = 0.12) -> list[str]:
        q_norm = normalize_join(query)
        if not q_norm.strip():
            return []

        v = self.vectorizer.transform([q_norm])
        sims = cosine_similarity(v, self.tfidf_matrix).flatten()

        idxs = sims.argsort()[-k:][::-1]
        out: list[str] = []
        for i in idxs:
            if sims[i] >= threshold:
                out.append(self.docs[i])
        return out

    def top_k_bm25(self, query: str, k: int = 3) -> list[str]:
        q_tokens = list(normalize(query))
        if not q_tokens:
            return []

        scores = np.array(self.bm25.get_scores(q_tokens), dtype=float)
        idxs = scores.argsort()[-k:][::-1]

        out: list[str] = []
        for i in idxs:
            if scores[i] > 0:
                out.append(self.docs[i])
        return out

    def top_k_hybrid(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.18,
        w_tfidf: float = 0.55,
        w_bm25: float = 0.45,
    ) -> list[str]:
        """
        Combina TF-IDF e BM25 normalizando ambos para 0..1 e somando com pesos.
        """
        q_norm = normalize_join(query)
        q_tokens = list(normalize(query))

        if (not q_norm.strip()) and (not q_tokens):
            return []

        # TF-IDF scores
        if q_norm.strip():
            v = self.vectorizer.transform([q_norm])
            tfidf_scores = cosine_similarity(v, self.tfidf_matrix).flatten()
        else:
            tfidf_scores = np.zeros(len(self.docs), dtype=float)

        # BM25 scores
        if q_tokens:
            bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=float)
        else:
            bm25_scores = np.zeros(len(self.docs), dtype=float)

        # Normaliza (0..1) com segurança
        def norm01(x: np.ndarray) -> np.ndarray:
            if x.size == 0:
                return x
            mn = float(np.min(x))
            mx = float(np.max(x))
            if mx - mn <= 1e-12:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        tfidf_n = norm01(tfidf_scores)
        bm25_n = norm01(bm25_scores)

        final = (w_tfidf * tfidf_n) + (w_bm25 * bm25_n)

        idxs = final.argsort()[-k:][::-1]
        out: list[str] = []
        for i in idxs:
            if final[i] >= threshold:
                out.append(self.docs[i])

        return out
