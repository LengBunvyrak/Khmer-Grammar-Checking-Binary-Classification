import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.config import POS_TAGS


POS_TO_ID = {tag: i for i, tag in enumerate(POS_TAGS)}
POS_UNKNOWN = POS_TO_ID["UNKNOWN"]


def pos_tags_to_ids(pos_tags):
    return [POS_TO_ID.get(tag, POS_UNKNOWN) for tag in pos_tags]


class KhmerTextDataset(Dataset):
    _embedding_cache = {}

    def __init__(self, tokens_list, pos_tags_list, labels, embedding_model, use_cache=True):
        self.tokens_list = tokens_list
        self.pos_tags_list = pos_tags_list
        self.labels = labels
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.get_dimension()
        self.use_cache = use_cache
        if self.use_cache:
            self._build_cache()

    def _build_cache(self):
        unique_tokens = set()
        for tokens in self.tokens_list:
            unique_tokens.update(tokens)
        new_tokens = unique_tokens - set(self._embedding_cache.keys())
        if new_tokens:
            print(f"Caching embeddings for {len(new_tokens)} new unique tokens...")
            for token in new_tokens:
                try:
                    vec = self.embedding_model.get_word_vector(token)
                except Exception:
                    vec = np.zeros(self.embedding_dim)
                self._embedding_cache[token] = vec
            print(f"Total cached tokens: {len(self._embedding_cache)}")

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        pos_tags = self.pos_tags_list[idx]
        label = self.labels[idx]
        embeddings = []
        for token in tokens:
            if self.use_cache and token in self._embedding_cache:
                vec = self._embedding_cache[token]
            else:
                try:
                    vec = self.embedding_model.get_word_vector(token)
                except Exception:
                    vec = np.zeros(self.embedding_dim)
            embeddings.append(vec)
        embeddings = torch.FloatTensor(np.array(embeddings))
        pos_ids = torch.LongTensor(pos_tags_to_ids(pos_tags))
        label = torch.LongTensor([label])
        return embeddings, pos_ids, label, len(tokens)

    @classmethod
    def clear_cache(cls):
        cls._embedding_cache.clear()
        print("Embedding cache cleared")


def collate_batch(batch):
    embeddings_list, pos_ids_list, labels_list, lengths_list = zip(*batch)
    padded_embeddings = pad_sequence(embeddings_list, batch_first=True)
    padded_pos_ids = pad_sequence(pos_ids_list, batch_first=True, padding_value=POS_UNKNOWN)
    labels = torch.cat(labels_list)
    lengths = torch.LongTensor(lengths_list)
    return padded_embeddings, padded_pos_ids, labels, lengths
