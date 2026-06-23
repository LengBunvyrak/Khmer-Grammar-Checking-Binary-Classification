import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class SemanticCoherence:
    CONTENT_POS = {"NN", "VB", "JJ", "RB", "CD"}

    POS_WEIGHTS = {
        ("NN", "VB"): 1.5,
        ("VB", "NN"): 1.5,
        ("JJ", "NN"): 1.2,
        ("NN", "NN"): 1.0,
        ("VB", "VB"): 1.0,
        ("RB", "VB"): 0.8,
    }

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.dim = embedding_model.get_dimension()

    def get_vector(self, word):
        try:
            return self.embedding_model.get_word_vector(word)
        except Exception:
            return np.zeros(self.dim)

    def get_vectors_batch(self, words):
        vectors = np.array([self.get_vector(word) for word in words])
        return vectors

    def weighted_semantic_coherence(self, tokens, pos_tags):
        if len(tokens) < 2:
            return 0.5
        vectors = self.get_vectors_batch(tokens)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_vectors = vectors / norms
        scores = []
        weights = []
        for i in range(len(tokens) - 1):
            current_pos = pos_tags[i]
            next_pos = pos_tags[i + 1]
            weight = self.POS_WEIGHTS.get((current_pos, next_pos), 0.5)
            if weight > 0:
                sim = np.dot(normalized_vectors[i], normalized_vectors[i + 1])
                scores.append(sim)
                weights.append(weight)
        if not scores:
            return 0.0
        return np.average(scores, weights=weights)

    def content_word_coherence(self, tokens, pos_tags):
        content_indices = [i for i in range(len(tokens)) if pos_tags[i] in self.CONTENT_POS]
        if len(content_indices) < 2:
            return 0.5
        content_tokens = [tokens[i] for i in content_indices]
        vectors = self.get_vectors_batch(content_tokens)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_vectors = vectors / norms
        scores = []
        for i in range(len(normalized_vectors) - 1):
            sim = np.dot(normalized_vectors[i], normalized_vectors[i + 1])
            scores.append(sim)
        return np.mean(scores)

    def multi_distance_coherence(self, tokens, max_distance=3):
        if len(tokens) < 2:
            return 0.5
        vectors = self.get_vectors_batch(tokens)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_vectors = vectors / norms
        all_scores = []
        for distance in range(1, min(max_distance + 1, len(tokens))):
            pairs_start = normalized_vectors[:-distance]
            pairs_end = normalized_vectors[distance:]
            similarities = np.sum(pairs_start * pairs_end, axis=1)
            weighted_sims = similarities / distance
            all_scores.extend(weighted_sims)
        return np.mean(all_scores) if all_scores else 0.0

    def ensemble_coherence_score(self, tokens, pos_tags):
        if len(tokens) < 2:
            return 0.5
        s1 = self.weighted_semantic_coherence(tokens, pos_tags)
        s2 = self.content_word_coherence(tokens, pos_tags)
        s3 = self.multi_distance_coherence(tokens)
        return 0.4 * s1 + 0.3 * s2 + 0.3 * s3

    def extract_features(self, df):
        print("Calculating semantic coherence features (vectorized)...")
        coherence_scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Semantic Analysis"):
            score = self.ensemble_coherence_score(row["tokens"], row["pos_tags"])
            coherence_scores.append(score)
        df["semantic_coherence"] = coherence_scores
        print(f"Mean coherence: {df['semantic_coherence'].mean():.3f}")
        return df
