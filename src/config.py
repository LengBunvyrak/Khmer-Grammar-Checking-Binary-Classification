import torch

SEED = 42
FASTTEXT_MODEL_PATH = "cc.km.300.bin"
DATA_PATH = "train_data.csv"
MODEL_SAVE_PATH = "gru_model.pth"

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = 2
N_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
N_EPOCHS = 15
BATCH_SIZE = 32
PATIENCE = 3
CLIP_VALUE = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLUMNS = [
    "oov_ratio",
    "dep_grammar_score",
    "has_complete_clause",
    "semantic_coherence",
    "grammar_oov_interaction",
]
