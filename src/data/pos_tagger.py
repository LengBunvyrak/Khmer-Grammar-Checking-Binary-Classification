from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class KhmerPOSTagger:
    def __init__(self, model_name="seanghay/khmer-pos-roberta"):
        print(f"Loading POS tagger: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )
        print("POS tagger loaded successfully")

    def tag_sentence(self, sentence):
        try:
            tags = self.pipeline(sentence)
            tokens = [t["word"] for t in tags]
            pos_tags = [t["entity_group"] for t in tags]
            return tokens, pos_tags
        except Exception as e:
            print(f"POS tagging error for sentence: {sentence[:50]}... Error: {e}")
            return [], []

    def tag_dataframe(self, df, text_column="text"):
        print(f"Tagging {len(df)} sentences...")
        tokens_list = []
        pos_tags_list = []
        for sentence in tqdm(df[text_column], desc="POS Tagging"):
            tokens, pos_tags = self.tag_sentence(sentence)
            tokens_list.append(tokens)
            pos_tags_list.append(pos_tags)
        df["tokens"] = tokens_list
        df["pos_tags"] = pos_tags_list
        print("POS tagging complete")
        print("Added 'tokens' and 'pos_tags' columns")
        return df
