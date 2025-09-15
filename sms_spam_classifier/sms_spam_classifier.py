# === SMS Spam Classifier ===
import numpy as np 
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import LabelEncoder # pyright: ignore[reportMissingModuleSource]
from sklearn.feature_extraction.text import TfidfVectorizer # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LogisticRegression # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import GradientBoostingClassifier # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import accuracy_score, precision_score, log_loss # pyright: ignore[reportMissingModuleSource]

# === 1) Load dataset ===
# Make sure you have "spam.csv" in your working directory
# Common format: label (ham/spam), message (text)
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]

df.columns = ["label", "message"]

# Encode labels: ham=0, spam=1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print("Dataset size:", df.shape)
print(df["label"].value_counts())

# === 2) Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

# === 3) Convert text to TF-IDF features ===
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 4) Baseline model: Logistic Regression ===
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train_vec, y_train)

y_pred = logreg.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)

print("\n=== Baseline (Logistic Regression) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision (macro): {prec_macro:.4f}")

# === 5) Gradient Boosting Classifier ===
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
gb.fit(X_train_vec.toarray(), y_train)

# Compute train/test log-loss across boosting stages
train_losses, test_losses = [], []
for proba_tr, proba_te in zip(
    gb.staged_predict_proba(X_train_vec.toarray()),
    gb.staged_predict_proba(X_test_vec.toarray())
):
    train_losses.append(log_loss(y_train, proba_tr, labels=np.unique(y_train)))
    test_losses.append(log_loss(y_test, proba_te, labels=np.unique(y_test)))

# Final metrics
y_pred_gb = gb.predict(X_test_vec.toarray())
acc_gb = accuracy_score(y_test, y_pred_gb)
prec_macro_gb = precision_score(y_test, y_pred_gb, average="macro", zero_division=0)

print("\n=== Gradient Boosting (Final) ===")
print(f"Accuracy : {acc_gb:.4f}")
print(f"Precision (macro): {prec_macro_gb:.4f}")

# === 6) Plot loss curves (train vs test) ===
plt.figure(figsize=(7,5))
plt.plot(train_losses, label="Train log-loss")
plt.plot(test_losses, label="Test log-loss")
plt.xlabel("Boosting iteration")
plt.ylabel("Log-loss")
plt.title("Training vs Test Loss (Gradient Boosting)")
plt.legend()
plt.tight_layout()
plt.show()







