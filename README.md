# IMDb Sentiment Analysis (NLP)

## Project Overview
This project builds an end-to-end **sentiment analysis system** to classify IMDb movie reviews as **positive or negative** using traditional NLP techniques and machine learning models.

The focus is not just on achieving good accuracy, but on building a **clean and explainable ML pipeline**.

---

## Problem Statement
Movie reviews contain rich natural language with mixed sentiment, negations, and subjective expressions. The goal is to design a system that can automatically determine whether a given review expresses **positive** or **negative** sentiment.

---

## Project Structure
```
sentiment-analysis-imdb/
│
├── data/
│   ├── raw/                 # Original IMDb dataset
│   └── cleaned/             # Preprocessed text + saved vectorizers
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_tfidf_vectorization.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   └── preprocess.py        # Reusable text preprocessing pipeline
│
└── README.md
```

---

## Data Exploration (EDA)
- Dataset contains **50,000 reviews**, evenly balanced between positive and negative classes
- Reviews vary significantly in length, with a long tail of very long reviews
- Raw text contains HTML tags, punctuation noise, and inconsistent formatting

**Conclusion:** preprocessing is essential before any modeling.

---

## Text Preprocessing
Implemented a reusable preprocessing pipeline (`src/preprocess.py`) that:
- Removes HTML tags
- Converts text to lowercase
- Removes punctuation and numbers
- Removes stopwords
- Applies lemmatization (noun-based, POS-agnostic)

*Lemmatization was intentionally conservative to preserve sentiment-bearing adjectives.*

---

## Feature Engineering: TF-IDF
Used **TF-IDF vectorization** to convert text into numerical features.

Two feature representations were evaluated:
- **Unigram TF-IDF** `(ngram_range=(1,1))`
- **Unigram + Bigram TF-IDF** `(ngram_range=(1,2))`

Both vectorizers were saved using `pickle` to ensure a frozen and reproducible feature space.

---

## Models Trained
The following models were trained and evaluated:

| Model | Features | Notes |
|-----|--------|------|
| Logistic Regression | Unigram | Strong baseline |
| Logistic Regression | Unigram + Bigram | Best overall |
| Naive Bayes | Unigram | Fast, decent baseline |
| Naive Bayes | Unigram + Bigram | Minor improvement |

---

## Results & Observations
- **Logistic Regression consistently outperformed Naive Bayes**
- Difference between unigram and bigram features was **small but consistent**
- IMDb reviews are long enough that unigrams already capture most sentiment
- Bigrams helped slightly with negations, but did not drastically change performance

Typical F1-score range: **~88–90%**

---

## Error Analysis
Error analysis was performed using the **original (raw) reviews**, not preprocessed text.

Common failure cases:
- Sarcasm and irony
- Mixed sentiment within long reviews
- Subtle negations ("not bad at all")
- Neutral language with sentiment implied by context

These errors reflect **semantic limitations of bag-of-words models**, not pipeline flaws.

---

## Key Takeaways
- Clean preprocessing + TF-IDF + Logistic Regression is a strong baseline for NLP tasks
- Feature engineering choices should be evaluated empirically, not assumed
- Proper train/test splitting with stratification is critical for fair evaluation
- Error analysis should always be done on **human-readable text**

---

## Tech Stack
- Python
- pandas, NumPy
- scikit-learn
- NLTK
- Jupyter Notebook

---

## Status
Project complete and reproducible

