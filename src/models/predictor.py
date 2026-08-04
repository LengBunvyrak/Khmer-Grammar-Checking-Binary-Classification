import pandas as pd
import torch
import numpy as np


class SentencePredictor:
    def __init__(self, feature_pipeline_instance, trained_model, scaler, feature_columns, model_type, threshold=0.5):
        self.feature_pipeline = feature_pipeline_instance
        self.model = trained_model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.model_type = model_type
        self.threshold = threshold
        self._cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model_type == "gru":
            self.model.to(self.device)

    def clear_cache(self):
        self._cache.clear()

    def extract_features_for_sentence(self, sentence, use_cache=True):
        if use_cache and sentence in self._cache:
            return self._cache[sentence]
        temp_df = pd.DataFrame({"text": [sentence], "sentence_correct": [0]})
        processed_df = self.feature_pipeline.extract_all_features(temp_df)
        if use_cache:
            self._cache[sentence] = processed_df
        return processed_df

    def predict_sentence(self, sentence, use_cache=True):
        processed_df = self.extract_features_for_sentence(sentence, use_cache=use_cache)
        features_dict = processed_df[self.feature_columns].iloc[0].to_dict()
        tokens = processed_df["tokens"].iloc[0]
        pos_tags = processed_df["pos_tags"].iloc[0]
        prediction_numeric = None
        confidence = None
        prediction_label = None
        if self.model_type == "ml":
            feature_vector = processed_df[self.feature_columns].iloc[0].values.reshape(1, -1)
            feature_vector = self.scaler.transform(feature_vector)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(feature_vector)[0]
                prediction_numeric = self.model.predict(feature_vector)[0]
                confidence = max(proba)
            else:
                prediction_numeric = self.model.predict(feature_vector)[0]
                confidence = None
        elif self.model_type == "gru":
            feature_vector = processed_df[self.feature_columns].iloc[0].values.reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            extra_features = torch.FloatTensor(feature_vector_scaled).to(self.device)
            temp_dataset = self.feature_pipeline.create_temp_dataset(tokens, pos_tags)
            self.model.eval()
            with torch.no_grad():
                embeddings, pos_ids, _, lengths = temp_dataset[0]
                embeddings = embeddings.unsqueeze(0).to(self.device)
                pos_ids = pos_ids.unsqueeze(0).to(self.device)
                lengths = torch.LongTensor([lengths]).to(self.device)
                if hasattr(self.model, "use_feature_fusion") and self.model.use_feature_fusion:
                    output, _ = self.model(embeddings, lengths, extra_features, pos_ids)
                else:
                    output = self.model(embeddings, lengths, pos_ids=pos_ids)
                probs = torch.softmax(output, dim=1)
                prediction_numeric = 1 if probs[0, 1].item() >= self.threshold else 0
                confidence = probs[0, prediction_numeric].item()
        prediction_label = "Right" if prediction_numeric == 1 else "Wrong"
        return {
            "sentence": sentence,
            "prediction": prediction_label,
            "prediction_numeric": prediction_numeric,
            "confidence": confidence,
            "features": features_dict,
            "tokens": tokens,
            "pos_tags": pos_tags,
        }

    def predict_batch(self, sentences, use_cache=True):
        results = []
        for sentence in sentences:
            result = self.predict_sentence(sentence, use_cache=use_cache)
            results.append(result)
        return results
