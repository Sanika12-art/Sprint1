import pandas as pd
import re
import string
import joblib

from nltk.corpus import stopwords

# sklearn libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1️⃣ Load Dataset
# =========================
df = pd.read_csv("dataset/cleaned_data.csv")

# Keep needed columns only
df = df[["text", "label"]]

# Remove null values
df = df.dropna(subset=["text", "label"])

# Convert text to string
df["text"] = df["text"].astype(str)

# Remove duplicate rows
df = df.drop_duplicates()

print("Dataset Loaded Successfully!")

# =========================
# 2️⃣ Clean Text
# =========================
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["text"] = df["text"].apply(clean_text)

# =========================
# 3️⃣ Define X and y
# =========================
X = df["text"]
y = df["label"]

# =========================
# 4️⃣ Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5️⃣ Pipeline
# =========================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression(max_iter=2000))
])

# =========================
# 6️⃣ GridSearchCV
# =========================
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [5000],
    "tfidf__min_df": [1, 2],
    "tfidf__max_df": [0.95],
    "tfidf__sublinear_tf": [True],
    "model__C": [0.1, 1, 10],
    "model__solver": ["liblinear"]
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

# =========================
# 7️⃣ Best Model
# =========================
best_model = grid.best_estimator_

# =========================
# 8️⃣ Evaluate Model
# =========================
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# =========================
# 9️⃣ Save Model
# =========================
joblib.dump(best_model.named_steps["model"], "model/fake_news_model.pkl")
joblib.dump(best_model.named_steps["tfidf"], "model/tfidf_vectorizer.pkl")

print("Model saved successfully!")