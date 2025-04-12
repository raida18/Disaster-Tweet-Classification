# 🌪️ Disaster Tweet Classification with Deep Learning

This project applies Natural Language Processing (NLP) and deep learning to classify whether a tweet is about a real disaster or not. Using a Gated Recurrent Unit (GRU) neural network and pretrained GloVe word embeddings, the model is trained to detect disaster-related content with an emphasis on F1-score performance.

---

## 🧠 Project Overview

The dataset comes from a Kaggle competition, where each tweet is labeled:
- `1` if it's **about a real disaster**
- `0` if it's **not related to a disaster**

The goal is to build a model that can accurately distinguish between these two classes.

---

## 📂 Dataset

- **Source:** [Kaggle - Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

---

## 🧪 Model Architecture

- **Tokenizer**: Keras tokenizer to convert tweets to integer sequences.
- **Embeddings**: 300-dimensional GloVe (Global Vectors for Word Representation).
- **Model**: Sequential GRU-based model with dropout for regularization.

### 🔧 Layers:
- Embedding Layer (with pretrained GloVe vectors)
- GRU Layer (64 units)
- Dropout Layer
- Dense Output Layer with Sigmoid activation

---

## 📊 Evaluation Metric

- **Custom F1 Score** implemented as the primary metric.
- **Why?** The dataset is slightly imbalanced, making F1 score more meaningful than plain accuracy.

---


