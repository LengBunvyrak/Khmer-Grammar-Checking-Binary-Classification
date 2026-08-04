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

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

FEATURE_COLUMNS = [
    "oov_ratio",
    "dep_grammar_score",
    "sentence_length",
    "pos_diversity",
    "avg_word_length",
]

USE_FEATURE_FUSION = True
NUM_EXTRA_FEATURES = len(FEATURE_COLUMNS)

# Heavier weight on Wrong (0) reduces over-predicting "Right" (the known false-positive bias).
CLASS_WEIGHTS = [1.3, 1.0]

# Per-token POS embedding (sequence grammar signal), fed into the GRU alongside word vectors.
POS_TAGS = [
    "AB", "AUX", "CC", "CD", "DBL", "DT", "ETC", "IN", "JJ", "KAN",
    "M", "NN", "PA", "PN", "PRO", "QT", "RB", "RPN", "SYM", "UH",
    "VB", "VB_JJ", "VCOM", "UNKNOWN",
]
POS_EMBEDDING_DIM = 32
