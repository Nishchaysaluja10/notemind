import math
from collections import defaultdict, Counter

class NgramAutocomplete:
    """A simple Trigram language model for query autocomplete."""
    def __init__(self):
        self.trigrams = defaultdict(Counter)
        
    def train(self, texts):
        for text in texts:
            words = text.lower().split()
            for i in range(len(words) - 2):
                context = (words[i], words[i+1])
                target = words[i+2]
                self.trigrams[context][target] += 1
                
    def suggest_next_word(self, query):
        words = query.lower().split()
        if len(words) < 2: return ""
        context = (words[-2], words[-1])
        if not self.trigrams[context]: return ""
        return self.trigrams[context].most_common(1)[0][0]

class CustomBM25:
    """BM25 keyword retrieval from scratch."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.df = defaultdict(int)
        self.avgdl = 0
        self.doc_lengths = []
        
    def fit(self, documents):
        self.documents = documents
        total_length = 0
        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            total_length += len(words)
            unique_words = set(words)
            for word in unique_words:
                self.df[word] += 1
        self.avgdl = total_length / len(documents) if documents else 0
        
    def search(self, query, top_k=3):
        query_words = query.lower().split()
        scores = []
        N = len(self.documents)
        for i, doc in enumerate(self.documents):
            score = 0
            doc_words = doc.lower().split()
            doc_counts = Counter(doc_words)
            for word in query_words:
                if word not in doc_counts: continue
                tf = doc_counts[word]
                idf = math.log(1 + (N - self.df[word] + 0.5) / (self.df[word] + 0.5))
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (self.doc_lengths[i] / self.avgdl)))
                score += idf * tf_norm
            scores.append((score, doc))
        return sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
