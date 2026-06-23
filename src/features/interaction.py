import pandas as pd


class InteractionFeatureExtractor:
    @staticmethod
    def calculate_grammar_oov_interaction(grammar_score, oov_ratio):
        return grammar_score * (1 - oov_ratio)

    def extract_features(self, df):
        print("Calculating interaction features...")
        df["grammar_oov_interaction"] = df.apply(
            lambda row: self.calculate_grammar_oov_interaction(
                row.get("dep_grammar_score", 1.0), row["oov_ratio"]
            ),
            axis=1,
        )
        print("Interaction features extracted")
        return df
