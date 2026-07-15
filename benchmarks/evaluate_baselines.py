import sys
import os
import random
import re
import numpy as np

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text.split()

# Vectorized Skip-gram Word2Vec using NumPy
class Word2VecTiny:
    def __init__(self, sentences, dim=992, window=5, lr=0.02, epochs=40):
        self.vocab = sorted(list(set([w for s in sentences for w in s])))
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.i2w = {i: w for i, w in enumerate(self.vocab)}
        self.V = len(self.vocab)
        self.dim = dim
        
        # Build training pairs
        pairs = []
        for sentence in sentences:
            indices = [self.w2i[w] for w in sentence]
            for idx, target in enumerate(indices):
                start = max(0, idx - window)
                end = min(len(indices), idx + window + 1)
                for ctx_idx in range(start, end):
                    if ctx_idx != idx:
                        pairs.append((target, indices[ctx_idx]))
                        
        pairs = np.array(pairs)
        M = len(pairs)
        
        np.random.seed(42)
        self.W_in = np.random.uniform(-0.1, 0.1, (self.V, self.dim))
        self.W_out = np.random.uniform(-0.1, 0.1, (self.dim, self.V))
        
        batch_size = 128
        for epoch in range(epochs):
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}...", flush=True)
            indices = np.random.permutation(M)
            for i in range(0, M, batch_size):
                batch_idx = indices[i:i+batch_size]
                targets = pairs[batch_idx, 0]
                contexts = pairs[batch_idx, 1]
                
                h = self.W_in[targets]
                u = np.dot(h, self.W_out)
                
                exp_u = np.exp(u - np.max(u, axis=1, keepdims=True))
                y = exp_u / np.sum(exp_u, axis=1, keepdims=True)
                
                e = y.copy()
                e[np.arange(len(targets)), contexts] -= 1.0
                
                dW_out = np.dot(h.T, e)
                dW_in = np.dot(e, self.W_out.T)
                
                self.W_out -= lr * dW_out
                for t, dw in zip(targets, dW_in):
                    self.W_in[t] -= lr * dw

    def get_similar(self, word, topn=12):
        if word not in self.w2i:
            return []
        idx = self.w2i[word]
        vec = self.W_in[idx]
        norms = np.linalg.norm(self.W_in, axis=1)
        norms[norms == 0] = 1e-8
        sims = np.dot(self.W_in, vec) / (norms * np.linalg.norm(vec))
        sims[idx] = -1.0
        sorted_indices = np.argsort(sims)[::-1][:topn]
        return [self.i2w[i] for i in sorted_indices]

# TF-IDF Cosine Similarity
class TFIDFTiny:
    def __init__(self, sentences):
        self.vocab = sorted(list(set([w for s in sentences for w in s])))
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.i2w = {i: w for i, w in enumerate(self.vocab)}
        self.V = len(self.vocab)
        self.N = len(sentences)
        
        tf = np.zeros((self.V, self.N))
        for d, s in enumerate(sentences):
            for w in s:
                tf[self.w2i[w], d] += 1
        
        df = np.sum(tf > 0, axis=1)
        idf = np.log((1 + self.N) / (1 + df)) + 1
        self.tfidf = tf * idf[:, np.newaxis]
        
    def get_similar(self, word, topn=12):
        if word not in self.w2i:
            return []
        idx = self.w2i[word]
        vec = self.tfidf[idx]
        norms = np.linalg.norm(self.tfidf, axis=1)
        norms[norms == 0] = 1e-8
        sims = np.dot(self.tfidf, vec) / (norms * np.linalg.norm(vec))
        sims[idx] = -1.0
        sorted_indices = np.argsort(sims)[::-1][:topn]
        return [self.i2w[i] for i in sorted_indices]

# Random
class RandomBaseline:
    def __init__(self, sentences):
        self.vocab = sorted(list(set([w for s in sentences for w in s])))
        
    def get_similar(self, word, topn=12):
        choices = [w for w in self.vocab if w != word]
        return random.sample(choices, min(len(choices), topn))

def main():
    corpus_path = "benchmarks/data/tinystories_sample.txt"
    if not os.path.exists(corpus_path):
        print(f"Error: {corpus_path} not found")
        sys.exit(1)
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if len(line.strip()) >= 24]
        
    # Test on 20 lines
    lines_20 = all_lines[:20]
    sentences_20 = [clean_and_tokenize(line) for line in lines_20]
    
    print("=== EVALUATION ON 20 LINES ===")
    w2v_20 = Word2VecTiny(sentences_20, dim=992, epochs=100)
    tfidf_20 = TFIDFTiny(sentences_20)
    rand_20 = RandomBaseline(sentences_20)
    
    test_words = ["robot", "car", "bear", "family"]
    for w in test_words:
        print(f"\nWord: {w}")
        print(f"  Random  : {', '.join(rand_20.get_similar(w))}")
        print(f"  TF-IDF  : {', '.join(tfidf_20.get_similar(w))}")
        print(f"  Word2Vec: {', '.join(w2v_20.get_similar(w))}")

if __name__ == "__main__":
    main()
