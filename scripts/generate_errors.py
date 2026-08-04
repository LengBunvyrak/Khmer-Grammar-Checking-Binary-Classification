"""
Generate a grammar-checking dataset from a raw Khmer sentence corpus.

Produces a balanced labeled CSV where:
  - label 1 = the original sentence (correct)
  - label 0 = a corrupted version, one of several error types

CRITICAL: corrupted sentences are written as the token list JOINED WITH NO
SPACES ("".join(tokens)), matching how Khmer is actually written. The previous
dataset wrote scrambled tokens space-separated, which made "contains a space"
a near-perfect Wrong detector (100% of Wrong rows had a space vs 49.9% of
Right rows) -- the classifier was memorizing spacing, not grammar.

Also propagates an optional `doc_id` column so you can split train/test by
source document and avoid leakage between a sentence and its corrupted twin.

Usage:
    # plain Khmer sentences, one per line; tagged with khmer-pos-roberta:
    python scripts/generate_errors.py corpus.txt -o data_augmented.csv
    # already-tagged CSV (columns: text,tokens,pos_tags[,doc_id]):
    python scripts/generate_errors.py tagged.csv --pretagged -o data_augmented.csv
"""

import argparse
import ast
import json
import random
from pathlib import Path

import pandas as pd

PARTS = [
    "គឺ", "ជា", "ក៏", "ដើម្បី", "និង", "ដែល", "បាន", "ទៅ", "មក", "នៅ",
]

SUBJECT_TAGS = ("NN", "PN", "PRO", "PR")
VERB_TAGS = ("VB", "VB_JJ", "AUX")


def error_shuffle(tokens):
    idx = list(range(len(tokens)))
    random.shuffle(idx)
    return [tokens[i] for i in idx]


def error_swap_adjacent(tokens):
    if len(tokens) < 2:
        return tokens
    i = random.randrange(len(tokens) - 1)
    out = list(tokens)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out


def error_drop_token(tokens):
    if len(tokens) <= 1:
        return tokens
    i = random.randrange(len(tokens))
    return [t for j, t in enumerate(tokens) if j != i]


def error_drop_subject(tokens, pos_tags):
    for i, tag in enumerate(pos_tags):
        if tag.startswith(SUBJECT_TAGS):
            return [t for j, t in enumerate(tokens) if j != i]
    return tokens


def error_front_verb(tokens, pos_tags):
    verb_i = next((i for i, t in enumerate(pos_tags) if t.startswith(VERB_TAGS)), None)
    if verb_i is None:
        return tokens
    out = list(tokens)
    out.insert(0, out.pop(verb_i))
    return out


def error_drop_particle(tokens, pos_tags):
    for i, tok in enumerate(tokens):
        if tok in PARTS:
            return [t for j, t in enumerate(tokens) if j != i]
    return tokens


def error_duplicate(tokens):
    if len(tokens) < 2:
        return tokens
    i = random.randrange(len(tokens))
    out = list(tokens)
    out.insert(i, out[i])
    return out


def error_fragment(tokens):
    if len(tokens) <= 2:
        return tokens
    return tokens[: random.randint(2, 3)]


ERROR_TYPES = [
    ("shuffle", error_shuffle),
    ("swap_adjacent", error_swap_adjacent),
    ("drop_token", error_drop_token),
    ("drop_subject", error_drop_subject),
    ("front_verb", error_front_verb),
    ("drop_particle", error_drop_particle),
    ("duplicate", error_duplicate),
    ("fragment", error_fragment),
]


def tag_corpus(texts):
    from src.data.pos_tagger import KhmerPOSTagger

    tagger = KhmerPOSTagger()
    tokens, pos_tags = [], []
    for text in texts:
        tok, pos = tagger.tag_sentence(text)
        tokens.append(tok)
        pos_tags.append(pos)
    return tokens, pos_tags


def make_row(text, tokens, pos_tags, label, error_type, doc_id):
    # join without spaces so spacing is never a predictive feature
    return {
        "text": "".join(tokens) if label == 0 else text,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "sentence_correct": label,
        "error_type": error_type,
        "doc_id": doc_id,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to corpus (text file or CSV).")
    ap.add_argument("--pretagged", action="store_true", help="CSV already has tokens+pos_tags columns.")
    ap.add_argument("-o", "--output", default="data_augmented.csv")
    ap.add_argument("--negatives-per-positive", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--doc-id-col", default="doc_id")
    args = ap.parse_args()
    random.seed(args.seed)

    path = Path(args.input)
    if path.suffix.lower() in (".csv", ".tsv") or args.pretagged:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "tokens" in df.columns:
            df["tokens"] = df["tokens"].apply(ast.literal_eval)
        if "pos_tags" in df.columns:
            df["pos_tags"] = df["pos_tags"].apply(ast.literal_eval)
        texts = df["text"].tolist()
        tokens = df["tokens"].tolist() if "tokens" in df.columns else None
        pos_tags = df["pos_tags"].tolist() if "pos_tags" in df.columns else None
        doc_ids = df[args.doc_id_col].tolist() if args.doc_id_col in df.columns else list(range(len(df)))
        if tokens is None or pos_tags is None:
            tokens, pos_tags = tag_corpus(texts)
    else:
        with open(path, encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
        tokens, pos_tags = tag_corpus(texts)
        doc_ids = list(range(len(texts)))

    rows = []
    for text, tok, pos, doc_id in zip(texts, tokens, pos_tags, doc_ids):
        if len(tok) < 3:
            continue
        rows.append(make_row(text, tok, pos, 1, "original", doc_id))
        candidates = list(ERROR_TYPES)
        random.shuffle(candidates)
        made = 0
        for name, fn in candidates:
            if made >= args.negatives_per_positive:
                break
            corrupt = fn(tok, pos) if name in ("drop_subject", "front_verb", "drop_particle") else fn(tok)
            if corrupt == tok or len(corrupt) == 0:
                continue
            rows.append(make_row(text, corrupt, pos, 0, name, doc_id))
            made += 1

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out):,} rows to {args.output}")
    print(out["sentence_correct"].value_counts().to_dict())
    print("\nError-type distribution:")
    print(out["error_type"].value_counts())


if __name__ == "__main__":
    main()
