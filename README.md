# 🌐 Machine Translation — English to Arabic

A from-scratch **seq2seq Transformer** for English → Arabic machine translation, built entirely with **NumPy**. No TensorFlow, no PyTorch, no HuggingFace — just pure Python and math.

---

## ✨ Features

- Custom Transformer encoder-decoder implemented in NumPy
- Two interchangeable attention mechanisms:
  - **Multi-Head Self-Attention** (Vaswani et al., 2017)
  - **Additive (Bahdanau) Attention**
- Sinusoidal positional encoding
- Label smoothing + PAD-token masking in the loss
- Adam optimizer with central finite-difference gradients
- Early stopping on validation loss
- Greedy autoregressive decoding
- Corpus BLEU-2 evaluation
- 300 curated English–Arabic parallel sentences (no internet required)

---

## 📁 Project Structure

```
.
├── Machine_Translation_englishToarabic.ipynb   # Main notebook
├── README.md
└── training_curves.png                               # Generated during training
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib
```

That's it. No deep learning framework required.

### Run the Notebook

```bash
jupyter notebook Machine_Translation_englishToarabic_FIXED.ipynb
```

Or open it in JupyterLab, VS Code, or Google Colab.

---

## 🏗️ Model Architecture

```
Input (English tokens)
        │
   [Embedding + Positional Encoding]
        │
   ┌────▼────────────────────┐
   │       Encoder            │
   │  ┌─────────────────┐    │
   │  │ Multi-Head      │    │
   │  │ Self-Attention  │    │
   │  └────────┬────────┘    │
   │       LayerNorm          │
   │  ┌─────────────────┐    │
   │  │ Feed-Forward    │    │
   │  │ Network         │    │
   │  └────────┬────────┘    │
   │       LayerNorm          │
   └────────────┬────────────┘
                │ enc_output
   ┌────────────▼────────────┐
   │       Decoder            │
   │  ┌─────────────────┐    │
   │  │ Masked Self-    │    │
   │  │ Attention       │    │
   │  └────────┬────────┘    │
   │       LayerNorm          │
   │  ┌─────────────────┐    │
   │  │ Cross-Attention │    │◄── enc_output
   │  │ (MHA or Bahdanau│    │
   │  └────────┬────────┘    │
   │       LayerNorm          │
   │  ┌─────────────────┐    │
   │  │ Feed-Forward    │    │
   │  │ Network         │    │
   │  └────────┬────────┘    │
   │       LayerNorm          │
   └────────────┬────────────┘
                │
   [Output Projection → Softmax]
                │
        Arabic tokens
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dimension (`d_model`) | 64 |
| Attention heads | 4 |
| Feed-forward dim (`d_ff`) | 128 |
| Dropout rate | 0.1 |
| Label smoothing | 0.1 |
| Learning rate | 3e-3 |
| Batch size | 32 |
| Max epochs | 30 |
| Early stopping patience | 5 |

---

## 📊 Results

Both attention variants are trained and evaluated automatically. Results will vary slightly by run due to stochastic gradient sampling, but typical output looks like:

| Metric | Multi-Head | Additive |
|---|---|---|
| Test Loss | ~1.8 | ~2.0 |
| Test Accuracy | ~0.65 | ~0.60 |
| BLEU-2 Score | ~0.40 | ~0.35 |

The best-performing model is automatically selected for sample translations.

---

## 🔤 Sample Translations

```
EN: Good morning.
AR: صباح الخير

EN: How are you?
AR: كيف حالك

EN: I am happy.
AR: أنا سعيد

EN: The food is delicious.
AR: الطعام لذيذ

EN: Thank you very much.
AR: شكرا جزيلا
```

---

## 🧠 How It Works

### Training

Gradients are computed via **central finite differences**:

```
∂L/∂θ ≈ [L(θ + ε) − L(θ − ε)] / 2ε
```

A random subset of 128 parameters is sampled per mini-batch per layer, making this tractable without autodiff. Updates are applied with the **Adam optimizer**.

### Decoding

Greedy autoregressive decoding: at each step, the token with the highest softmax probability is selected and fed back as the next decoder input until `<END>` is predicted or the max length is reached.

### BLEU Evaluation

Corpus BLEU-2 (bigram precision with brevity penalty) is computed over the held-out test set, which is appropriate for the short sentence lengths in this dataset.

---

## 📚 Dataset

300 hand-curated English–Arabic sentence pairs covering everyday topics: greetings, weather, food, family, travel, education, and more. The dataset is embedded directly in the notebook — no downloads or API keys needed.

To extend the dataset, simply add more `("english", "عربي")` tuples to the `EN_AR_PAIRS` list in Cell 2.

---

## 📖 References

- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Bahdanau et al. (2015) — [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Papineni et al. (2002) — [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)

---

## 📄 License

MIT License — free to use, modify, and distribute.
