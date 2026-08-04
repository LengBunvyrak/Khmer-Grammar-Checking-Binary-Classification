"""Compile the project's .py sources into a self-contained, runnable notebook.

Each module is written to the runtime filesystem with %%writefile, then the
scripts are executed. Output: Khmer_Grammar_Checker_Improved.ipynb

Run: python scripts/build_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Khmer_Grammar_Checker_Improved.ipynb"

MODULES = [
    ("src/__init__.py", ""),
    ("src/config.py", ""),
    ("src/utils.py", ""),
    ("src/data/__init__.py", ""),
    ("src/data/pos_tagger.py", ""),
    ("src/data/dataset.py", ""),
    ("src/features/__init__.py", ""),
    ("src/features/oov.py", ""),
    ("src/features/grammar.py", ""),
    ("src/features/structural.py", ""),
    ("src/models/__init__.py", ""),
    ("src/models/gru.py", ""),
    ("src/models/train_utils.py", ""),
    ("src/models/predictor.py", ""),
    ("src/pipeline/__init__.py", ""),
    ("src/pipeline/feature_pipeline.py", ""),
    ("train.py", ""),
    ("evaluate.py", ""),
    ("predict.py", ""),
    ("train_transformer.py", ""),
    ("scripts/generate_errors.py", ""),
]

MARKDOWN = """\
# Khmer Grammar Checking — Improved Binary Classification

ITM 454 NLP course project. Two model paths:

1. **BiGRU + attention + feature fusion** (`train.py`): FastText word vectors
   concatenated with per-token POS embeddings, plus 5 scalar features.
   Weighted loss + tuned decision threshold to counter the false-positive bias.
2. **Khmer transformer** (`train_transformer.py`): fine-tunes
   `seanghay/xlm-roberta-khmer-small` directly on raw text.

The old dataset had a **"space" leak**: 100% of Wrong rows contained a space
(space-separated scrambled tokens) vs 49.9% of Right rows. A model could cheat
on spacing alone. See `scripts/generate_errors.py` for the fixed data generator
(negatives joined without spaces) and run the data-quality cell below to verify.
Full change log: `CHANGES.md`. Backup of the pre-change project: `../KGCBC`.
"""

SETUP_MD = """\
## 1. Setup

Install dependencies and download the Khmer FastText model if missing.
In Colab, make sure `train_data.csv` and this notebook are in the working
directory (upload them, or mount Drive and `%cd` into the folder).
"""

SETUP_CODE = """\
# !pip install -r requirements.txt
# !wget -q https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.km.300.bin.gz
# !gunzip cc.km.300.bin.gz
"""

WRITE_MD = """\
## 2. Write the project modules

Each source file is written to the runtime filesystem verbatim so the
scripts can run unchanged.
"""

CHECK_MD = """\
## 3. Data quality check — the "space" leak

Verify the artifact that motivated the data overhaul. A healthy dataset should
show roughly the same spacing pattern for both classes.
"""

CHECK_CODE = """\
import csv

rows = list(csv.reader(open('train_data.csv', encoding='utf-8-sig')))[1:]

def has_space(t):
    return ' ' in t

sp = {0: 0, 1: 0}
tot = {0: 0, 1: 0}
for r in rows:
    lab = int(r[3])
    tot[lab] += 1
    sp[lab] += has_space(r[0])

print('RIGHT: %d rows, %d contain a space (%.1f%%)' % (tot[1], sp[1], 100 * sp[1] / tot[1]))
print('WRONG: %d rows, %d contain a space (%.1f%%)' % (tot[0], sp[0], 100 * sp[0] / tot[0]))
"""

TRAIN_MD = """\
## 4. Train the BiGRU

Splits data, trains with feature fusion + POS embeddings + class weights,
tunes the decision threshold on validation, saves `gru_model.pth`.
"""

EVAL_MD = """\
## 5. Evaluate

Loads the checkpoint, applies the saved threshold, prints a correct
`Wrong/Right` report + confusion matrix.
"""

PREDICT_MD = """\
## 6. Try it on single sentences

One clearly correct sentence and one scrambled (incorrect) sentence.
"""

PREDICT_CODE = """\
# A clearly correct sentence, then a scrambled (incorrect) one:
!python predict.py "កុមារកម្ពុជាទៅសាលារៀននៅព្រឹកនេះ"
!python predict.py "បាន ស៊ុត នាទីទី ដំបូង ពាក់កណ្តាល សំរាប់ អ៊ីតាលី"
"""

TF_MD = """\
## 7. (Optional) Fine-tune the Khmer transformer

Highest expected accuracy ceiling. Trains on raw text — no FastText or
feature pipeline. Saves `transformer_model.pth`. Needs the `transformers`
model download (~200MB); skip this cell to keep runtime short.
"""

DATA_MD = """\
## 8. (Optional) Generate a new dataset with the error taxonomy

Feed a raw Khmer corpus (one sentence per line). Corruptions are written
space-free so spacing can never become a predictive feature.
"""

TF_CODE = """\
# Optional: downloads the model + trains 3 epochs. Comment out to skip.
!python train_transformer.py
"""


def cell(code, cell_type="code"):
    return {"cell_type": cell_type, "metadata": {}, "source": code, "outputs": []}


def write_cell(rel_path, content):
    return cell("%%writefile %s\n%s" % (rel_path, content))


def build():
    cells = []
    cells.append(cell(MARKDOWN, "markdown"))
    cells.append(cell(SETUP_MD, "markdown"))
    cells.append(cell(SETUP_CODE))
    cells.append(cell(WRITE_MD, "markdown"))
    cells.append(cell("!mkdir -p src/data src/features src/models src/pipeline scripts"))
    for rel_path, _ in MODULES:
        cells.append(write_cell(rel_path, (ROOT / rel_path).read_text(encoding="utf-8")))
    cells.append(cell(CHECK_MD, "markdown"))
    cells.append(cell(CHECK_CODE))
    cells.append(cell(TRAIN_MD, "markdown"))
    cells.append(cell("!python train.py"))
    cells.append(cell(EVAL_MD, "markdown"))
    cells.append(cell("!python evaluate.py"))
    cells.append(cell(PREDICT_MD, "markdown"))
    cells.append(cell(PREDICT_CODE))
    cells.append(cell(TF_MD, "markdown"))
    cells.append(cell(TF_CODE))
    cells.append(cell(DATA_MD, "markdown"))
    cells.append(cell("!python scripts/generate_errors.py --help"))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
