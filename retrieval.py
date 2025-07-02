import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from preprocessing import normalize

class Retriever:
    def __init__(self, sentences: list[str],
                 ngram_range=(1, 2), max_df=0.85, min_df=2):
        self.sentences = sentences
        self.vectorizer = TfidfVectorizer(
            tokenizer=normalize,
            lowercase=False,
            stop_words=None,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(sentences)
        tokens = [normalize(s) for s in sentences]
        self.bm25 = BM25Okapi(tokens)

    def save(self, path: str):
        joblib.dump((self.vectorizer, self.tfidf_matrix, self.sentences, self.bm25), path)

    @staticmethod
    def load(path: str) -> 'Retriever':
        vectorizer, tfidf_matrix, sentences, bm25 = joblib.load(path)
        ret = Retriever.__new__(Retriever)
        ret.vectorizer = vectorizer
        ret.tfidf_matrix = tfidf_matrix
        ret.sentences = sentences
        ret.bm25 = bm25
        return ret

    def top_k_tfidf(self, query: str, k: int = 3, threshold: float = 0.1) -> list[str]:
        v = self.vectorizer.transform([query])
        sims = cosine_similarity(v, self.tfidf_matrix).flatten()
        idxs = sims.argsort()[-k:][::-1]
        return [self.sentences[i] for i in idxs if sims[i] >= threshold]

    def top_k_bm25(self, query: str, k: int = 3) -> list[str]:
        scores = self.bm25.get_scores(normalize(query))
        idxs = scores.argsort()[-k:][::-1]
        return [self.sentences[i] for i in idxs if scores[i] > 0]
