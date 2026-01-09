import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocess import clean_text


# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv("data/cleaned/imdb_cleaned.csv")

X_text = df["clean_review"]
y = df["label"]   # assume 1 = positive, 0 = negative


# -----------------------------
# Load saved TF-IDF vectorizer
# -----------------------------
with open("data/cleaned/tfidf_bigram.pkl", "rb") as f:
    vectorizer = pickle.load(f)

X = vectorizer.transform(X_text)


# -----------------------------
# Train Logistic Regression
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

model.fit(X, y)


# -----------------------------
# Save trained model
# -----------------------------
with open("data/cleaned/logreg_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Logistic Regression model saved successfully.")
