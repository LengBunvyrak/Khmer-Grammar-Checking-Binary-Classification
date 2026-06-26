import pandas as pd
from src.utils import load_fasttext_model
from src.data.pos_tagger import KhmerPOSTagger
from src.features.oov import EmbeddingOOVCalculator
from src.features.grammar import SimplePOSGrammarExtractor
from src.features.structural import StructuralFeatureExtractor


class FeaturePipeline:
    def __init__(self, embedding_model_path=None, embedding_model=None, lazy_pos=True):
        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif embedding_model_path:
            self.embedding_model = load_fasttext_model(embedding_model_path)
        else:
            raise ValueError("Provide embedding_model or embedding_model_path")
        self.lazy_pos = lazy_pos
        self.pos_tagger = None if lazy_pos else KhmerPOSTagger()
        self.oov_extractor = EmbeddingOOVCalculator(self.embedding_model)
        self.grammar_extractor = SimplePOSGrammarExtractor()
        self.structural_extractor = StructuralFeatureExtractor()

    def _ensure_pos_tagger(self):
        if self.pos_tagger is None:
            self.pos_tagger = KhmerPOSTagger()

    def extract_all_features(self, df):
        if "text" not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column for POS tagging.")
        self._ensure_pos_tagger()
        df = self.pos_tagger.tag_dataframe(df)
        df = self.oov_extractor.add_oov_features(df)
        df = self.grammar_extractor.extract_features(df)
        df = self.structural_extractor.extract_features(df)
        return df

    def create_temp_dataset(self, tokens):
        from src.data.dataset import KhmerTextDataset

        return KhmerTextDataset([tokens], [0], self.embedding_model, use_cache=True)

    @staticmethod
    def get_feature_cols():
        return [
            "oov_ratio",
            "dep_grammar_score",
            "sentence_length",
            "pos_diversity",
            "avg_word_length",
        ]
