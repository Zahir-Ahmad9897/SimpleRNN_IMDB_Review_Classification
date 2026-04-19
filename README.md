# 🎬 IMDB Sentiment Classification with SimpleRNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red?logo=keras)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A binary sentiment classifier built with a **Simple Recurrent Neural Network (SimpleRNN)** trained on the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). Given a plain-text movie review, the model predicts whether the sentiment is **Positive** or **Negative** with a confidence score.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Usage Example](#usage-example)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project demonstrates end-to-end text sentiment classification using a **SimpleRNN** architecture implemented in TensorFlow/Keras. The entire training pipeline was originally run on **Google Colab** (GPU: NVIDIA T4). The trained model is then served via a **Streamlit** web application that accepts free-form text and returns a sentiment label with a probability score.

**Task:** Binary Classification (`Positive` / `Negative`)  
**Dataset:** [IMDB Movie Reviews](https://keras.io/api/datasets/imdb/) — 50,000 reviews (25k train / 25k test)  
**Framework:** TensorFlow 2.x / Keras

---

## Project Structure

```
SimpleRNN_IMDB_Review_Classification/
│
├── SimpleRNN.ipynb          # Training notebook (Google Colab)
├── Prediction.ipynb         # Inference / exploration notebook
├── app.py                   # Streamlit web application (local)
├── rnn_app.py               # Original Colab deployment script
├── Simple_RNN_IMDB.keras    # Saved model (Keras native format) ← primary
├── Simple_RNN_IMDB.h5       # Saved model (HDF5 legacy format)
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

## Model Architecture

| Layer         | Type       | Output Shape    | Parameters  |
|---------------|------------|-----------------|-------------|
| Embedding     | Embedding  | (None, 500, 128)| 1,280,000   |
| SimpleRNN     | SimpleRNN  | (None, 128)     | 32,896      |
| Dense         | Dense      | (None, 1)       | 129         |

**Total Parameters:** 1,313,025 (~5.01 MB)

```
Input (integer sequences, max_len=500)
    ↓
Embedding(vocab_size=10000, embed_dim=128)
    ↓
SimpleRNN(units=128, activation='relu')
    ↓
Dense(1, activation='sigmoid')
    ↓
Output: Probability [0, 1]  →  Negative (<0.5) / Positive (≥0.5)
```

---

## Training Details

| Hyperparameter      | Value                 |
|---------------------|-----------------------|
| Vocabulary size     | 10,000 most frequent words |
| Sequence max length | 500 tokens (pre-padded) |
| Embedding dimension | 128                   |
| RNN units           | 128                   |
| Optimizer           | Adam                  |
| Loss function       | Binary Cross-Entropy  |
| Batch size          | 32                    |
| Max epochs          | 10                    |
| Validation split    | 20%                   |
| Early stopping      | patience=5, monitor=val_loss, restore_best_weights=True |
| Hardware            | Google Colab — NVIDIA T4 GPU |

---

## Results

Training logs (best epoch before early stopping):

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|-------|---------------|------------|--------------|----------|
| 1     | 85.49%        | 0.3736     | 76.10%       | 0.5148   |
| 2     | 87.74%        | 0.3380     | 76.10%       | 0.5080   |
| 3     | 88.91%        | 0.3182     | 75.92%       | 0.5055   |
| 4     | 90.11%        | 0.9869     | 76.30%       | **0.5031** |
| 5     | 90.77%        | 0.2765     | 76.70%       | 11.634   |

> **Validation accuracy ~77%** — Characteristic of `SimpleRNN` which suffers from the vanishing gradient problem on long sequences. Consider upgrading to LSTM/GRU for ~85–90% accuracy.

**Sample Inference:**
```
Review  : "The movie was fantastic! the acting was great and the plot was thrilling."
Sentiment: Positive
Score   : 0.9993 (99.93% confidence)
```

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip

### 1. Clone the repository
```bash
git clone https://github.com/Zahir-Ahmad9897/SimpleRNN_IMDB_Review_Classification.git
cd SimpleRNN_IMDB_Review_Classification
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

> **Note:** The pre-trained model files (`Simple_RNN_IMDB.keras` / `Simple_RNN_IMDB.h5`) must be present in the project root directory.

---

## Usage Example

Once the Streamlit app is running:

1. Type or paste any movie review into the text area.
2. Click **"Classify Sentiment"**.
3. The app will display:
   - **Sentiment**: `Positive` ✅ or `Negative` ❌
   - **Confidence Score**: A probability between 0 and 1

**Example reviews to try:**

```
✅ Positive:
"An absolutely breathtaking film. The performances were stellar
and the cinematography was stunning throughout."

❌ Negative:
"This movie was a complete waste of time. Terrible acting,
boring plot, and no character development whatsoever."
```

---

## Known Limitations

- **Vanishing Gradient:** `SimpleRNN` struggles with long sequences (>100 tokens). The validation accuracy ~77% reflects this limitation.
- **OOV Words:** Words outside the top 10,000 most frequent are treated as unknown tokens.
- **No Negation Handling:** Phrases like "not good" may be misclassified since the model lacks contextual depth.
- **Colab-specific artifacts:** `rnn_app.py` contains Colab magic commands (`!pip`, `!streamlit`, localtunnel) — use `app.py` for local deployment instead.

---

## Future Improvements

- [ ] Replace `SimpleRNN` with `LSTM` or `GRU` to address vanishing gradients (~85–90% accuracy)
- [ ] Add `Dropout` and `Recurrent Dropout` layers for better regularization
- [ ] Use pre-trained word embeddings (GloVe / Word2Vec / FastText)
- [ ] Add a training visualization (loss/accuracy curves) to the Streamlit app
- [ ] Containerize the application with Docker for portable deployment
- [ ] Add unit tests for preprocessing and prediction functions

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Zahir Ahmad**  
📧 GitHub: [@Zahir-Ahmad9897](https://github.com/Zahir-Ahmad9897)

> *Built as a hands-on deep learning project to practice RNN-based NLP classification.*