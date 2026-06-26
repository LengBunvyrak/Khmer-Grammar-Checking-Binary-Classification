import pandas as pd


class SimplePOSGrammarExtractor:
    @staticmethod
    def calculate_grammar_score(pos_tags):
        if not pos_tags or len(pos_tags) == 0:
            return 0.0

        score = 0.0

        has_noun = any(t.startswith("NN") for t in pos_tags)
        has_pronoun = any(t.startswith("PR") for t in pos_tags)
        has_verb = any(t.startswith("VB") for t in pos_tags)
        has_aux = any(t == "AUX" for t in pos_tags)
        has_adj = any(t.startswith("JJ") for t in pos_tags)
        has_adv = any(t.startswith("RB") for t in pos_tags)
        has_conj = any(t == "CC" for t in pos_tags)
        has_det = any(t.startswith("DT") for t in pos_tags)
        has_num = any(t.startswith("CD") for t in pos_tags)
        has_adp = any(t.startswith("IN") for t in pos_tags)

        has_subject = has_noun or has_pronoun
        has_predicate = has_verb or has_aux

        if has_subject and has_predicate:
            score += 0.45
        elif has_predicate:
            score += 0.15
        elif has_subject and not has_predicate:
            score -= 0.1

        if has_aux:
            score += 0.1
        if has_adj:
            score += 0.05
        if has_adv:
            score += 0.05
        if has_conj:
            score += 0.1
        if has_det or has_num or has_adp:
            score += 0.05

        unique_pos_count = len(set(pos_tags))
        score += min(0.1, unique_pos_count * 0.025)

        if not has_verb and not has_aux:
            score -= 0.25

        if len(pos_tags) <= 2:
            score -= 0.2

        if has_noun and not has_verb and not has_aux and not has_adj:
            score -= 0.2

        return max(0.0, min(1.0, score))

    @staticmethod
    def has_complete_clause(pos_tags):
        if not pos_tags or len(pos_tags) == 0:
            return 0
        has_noun = any(tag.startswith("NN") or tag.startswith("PR") for tag in pos_tags)
        has_verb = any(tag.startswith("VB") or tag == "AUX" for tag in pos_tags)
        if has_noun and has_verb:
            return 1
        return 0

    def extract_features(self, df):
        print("Extracting improved POS-based grammar features...")
        df["dep_grammar_score"] = df["pos_tags"].apply(self.calculate_grammar_score)
        print(f"Mean grammar score: {df['dep_grammar_score'].mean():.3f}")
        print(f"Grammar score std:  {df['dep_grammar_score'].std():.3f}")
        return df
