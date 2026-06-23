import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm


class EmbeddingOOVCalculator:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.vocab = set()
        if embedding_model:
            self._load_vocab_from_model()

    def _load_vocab_from_model(self):
        try:
            if hasattr(self.embedding_model, "get_words"):
                self.vocab = set(self.embedding_model.get_words())
            else:
                raise AttributeError("Model doesn't have recognized vocabulary interface")
            print(f"Loaded {len(self.vocab):,} words from embedding model")
        except Exception as e:
            print(f"Error loading vocabulary from model: {e}")

    def is_in_vocabulary(self, word):
        return word.strip() in self.vocab

    def calculate_oov_ratio(self, tokens):
        if isinstance(tokens, str):
            words = tokens.split()
        else:
            words = tokens
        if not words:
            return 0.0
        oov_count = sum(1 for word in words if not self.is_in_vocabulary(word))
        return oov_count / len(words)

    def calculate_oov_details(self, tokens):
        if isinstance(tokens, str):
            words = tokens.split()
        else:
            words = tokens
        oov_words = []
        known_words = []
        for word in words:
            if self.is_in_vocabulary(word):
                known_words.append(word)
            else:
                oov_words.append(word)
        return {
            "total_words": len(words),
            "known_words": len(known_words),
            "oov_words": len(oov_words),
            "oov_ratio": len(oov_words) / max(len(words), 1),
            "oov_word_list": oov_words,
            "known_word_list": known_words,
            "vocabulary_coverage": len(known_words) / max(len(words), 1),
        }

    def add_oov_features(self, df, text_col="tokens"):
        df = df.copy()
        print(f"Calculating OOV features for {len(df):,} texts...")
        oov_ratios = []
        vocab_coverages = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="OOV Analysis"):
            text = row[text_col]
            details = self.calculate_oov_details(text)
            oov_ratios.append(details["oov_ratio"])
            vocab_coverages.append(details["vocabulary_coverage"])
        df["oov_ratio"] = oov_ratios
        df["vocab_coverage"] = vocab_coverages
        print(f"Mean OOV ratio: {df['oov_ratio'].mean():.3f}")
        print(f"Mean vocab coverage: {df['vocab_coverage'].mean():.3f}")
        return df

    def extract_unknown_words(self, df, text_col="tokens", output_file="unknown_words.csv"):
        print("Extracting unknown words...")
        unknown_words = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Unknown Words"):
            text = row[text_col]
            details = self.calculate_oov_details(text)
            unknown_words.extend(details["oov_word_list"])
        word_counts = Counter(unknown_words)
        unknown_df = pd.DataFrame(
            [{"word": word, "frequency": count} for word, count in word_counts.most_common()]
        )
        print(f"Found {len(unknown_df):,} unique unknown words")
        if output_file and len(unknown_df) > 0:
            unknown_df.to_csv(output_file, index=False, encoding="utf-8")
            print(f"Saved to {output_file}")
        return unknown_df

    def evaluate_text_quality(self, text):
        details = self.calculate_oov_details(text)
        vocab_score = details["vocabulary_coverage"] * 100
        if vocab_score >= 95:
            quality = "Excellent"
            message = "Vocabulary is excellent"
        elif vocab_score >= 85:
            quality = "Good"
            message = "Good vocabulary usage"
        elif vocab_score >= 70:
            quality = "Fair"
            message = "Some vocabulary issues detected"
        else:
            quality = "Poor"
            message = "Many unknown words detected"
        return {
            "quality": quality,
            "vocab_score": vocab_score,
            "message": message,
            "total_words": details["total_words"],
            "known_words": details["known_words"],
            "unknown_words": details["oov_words"],
            "unknown_word_list": details["oov_word_list"],
        }
