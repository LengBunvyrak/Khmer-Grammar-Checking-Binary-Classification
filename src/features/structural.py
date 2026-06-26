import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class StructuralFeatureExtractor:
    @staticmethod
    def sentence_length(tokens):
        if isinstance(tokens, list):
            return len(tokens)
        return 0

    @staticmethod
    def pos_diversity(pos_tags):
        if not pos_tags or len(pos_tags) == 0:
            return 0.0
        return len(set(pos_tags)) / len(pos_tags)

    @staticmethod
    def avg_word_length(tokens):
        if not tokens or len(tokens) == 0:
            return 0.0
        lengths = [len(str(w)) for w in tokens]
        return np.mean(lengths)

    def extract_features(self, df):
        print("Extracting structural features...")
        df["sentence_length"] = df["tokens"].apply(self.sentence_length)
        df["pos_diversity"] = df["pos_tags"].apply(self.pos_diversity)
        df["avg_word_length"] = df["tokens"].apply(self.avg_word_length)
        print(f"Mean sentence length: {df['sentence_length'].mean():.2f}")
        print(f"Mean POS diversity:   {df['pos_diversity'].mean():.3f}")
        print(f"Mean word length:     {df['avg_word_length'].mean():.2f}")
        return df
