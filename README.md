# Khmer-Grammar-Checking-Binary-Classification
This is project part of the ITM 454 - Natural Language Processing course. In this repository contained a Khmer Grammar Checking System formulated as a binary sentence-level classification problem. Given an input Khmer sentence, the model will predicts whether the sentence is grammatically acceptable or unacceptable.

The system follow a stnadard supervised NLP pipeline adapted for low reosuce langauge setting. Text data are tokenized and POS tagged with the [khmer-pos-roberta](https://huggingface.co/seanghay/khmer-pos-roberta). The preprocessed sentenes are then process to extract feature with manual feature extraction method.

The classfication model is implemented using modern machine-learning and deep-learning frameworks. Feature representation include sentence-based feature, word embedding and grammatical feature to ensure the most optimal performance of the model. Training is performed using labeled data, optimized with cross-entropy loss and evaluate using standard classification metrics such as accuracy, precision, recall and F1 score.


