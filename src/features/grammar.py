import pandas as pd


class SimplePOSGrammarExtractor:
    @staticmethod
    def calculate_grammar_score(pos_tags):
        if len(pos_tags) < 2:
            return 0.5
        score = 0.5
        has_subject = any(tag.startswith("NN") for tag in pos_tags)
        has_verb = any(tag.startswith("VB") for tag in pos_tags)
        has_object = len([tag for tag in pos_tags if tag.startswith("NN")]) > 1
        if has_subject and has_verb:
            score += 0.3
        if has_object:
            score += 0.2
        return min(score, 1.0)

    @staticmethod
    def has_complete_clause(pos_tags):
        if not pos_tags or len(pos_tags) == 0:
            return 0
        has_noun = any(tag.startswith("NN") or tag.startswith("PR") for tag in pos_tags)
        has_verb = any(tag.startswith("VB") or tag == "AUX" for tag in pos_tags)
        has_aux = any(tag == "AUX" for tag in pos_tags)
        if has_noun and has_verb:
            return 1
        if has_verb and len(pos_tags) >= 1:
            if pos_tags[0].startswith("VB"):
                return 1
            for i in range(len(pos_tags) - 1):
                if pos_tags[i].startswith("VB") and pos_tags[i + 1].startswith("NN"):
                    return 1
        existential_markers = {"VB", "AUX", "CC"}
        has_existential = any(tag in existential_markers for tag in pos_tags)
        if has_existential and has_noun:
            return 1
        if "WP" in pos_tags or "WDT" in pos_tags or "WRB" in pos_tags:
            if has_verb:
                return 1
        if "CC" in pos_tags and has_verb:
            return 1
        if has_aux:
            return 1
        if any(tag.startswith("WP") or tag == "WDT" for tag in pos_tags) and has_verb:
            return 1
        verb_count = sum(1 for tag in pos_tags if tag.startswith("VB") or tag == "AUX")
        if verb_count >= 2:
            return 1
        has_adj = any(tag.startswith("JJ") for tag in pos_tags)
        if has_noun and has_adj and len(pos_tags) >= 2:
            for i in range(len(pos_tags) - 1):
                if pos_tags[i].startswith("NN") and pos_tags[i + 1].startswith("JJ"):
                    return 1
        only_nouns = has_noun and not has_verb and not has_adj
        only_verbs = has_verb and not has_noun and len(pos_tags) < 2
        if only_nouns or only_verbs:
            return 0
        return 1 if has_verb else 0

    def extract_features(self, df):
        print("Extracting simple POS-based grammar features...")
        df["dep_grammar_score"] = df["pos_tags"].apply(self.calculate_grammar_score)
        df["has_complete_clause"] = df["pos_tags"].apply(self.has_complete_clause)
        print(f"Mean grammar score: {df['dep_grammar_score'].mean():.3f}")
        print(
            f"Complete clauses: {df['has_complete_clause'].sum()} / {len(df)} "
            f"({df['has_complete_clause'].mean() * 100:.1f}%)"
        )
        return df
