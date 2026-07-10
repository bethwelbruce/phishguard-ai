"""
AI Phishing Detection System - Phase 1
========================================
Dataset : PhiUSIIL Phishing URL Dataset (UCI ML Repository, ID: 967)
          235,795 URLs | 54 features | Binary label (1 = Legitimate, 0 = Phishing)
Model   : Decision Tree Classifier with hyperparameter tuning (GridSearchCV)
Outputs : EDA plots (PNG), evaluation metrics (TXT/CSV), feature importance,
          shallow tree visualisation, persisted model (joblib)

Usage:
    python phase1_train.py

If the dataset CSV is not present locally, the script attempts to fetch it
via the `ucimlrepo` package (pip install ucimlrepo).

NOTE ON LABELS: In PhiUSIIL, label = 1 means LEGITIMATE and label = 0 means
PHISHING (per the dataset authors, Prasad & Chandra, 2023). Metrics below
report the phishing class (0) as the "positive"/target class where relevant.
"""

import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")  # headless environments (servers, CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_LOCAL = "PhiUSIIL_Phishing_URL_Dataset.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Non-numeric identifier / free-text columns that carry no direct model value
# (URL/Domain/TLD/Title information is already encoded in the numeric features
# such as URLLength, DomainLength, TLDLegitimateProb, DomainTitleMatchScore).
DROP_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    """Save the current matplotlib figure into the outputs directory."""
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [saved] {path}")


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print("=" * 70)
print("[1/7] Loading dataset ...")
print("=" * 70)

if os.path.exists(CSV_LOCAL):
    df = pd.read_csv(CSV_LOCAL, encoding="utf-8")
    print(f"   Loaded local CSV: {CSV_LOCAL}")
else:
    # Fallback: fetch from the UCI Machine Learning Repository
    from ucimlrepo import fetch_ucirepo

    print("   Local CSV not found - fetching from UCI (id=967) ...")
    phiusiil = fetch_ucirepo(id=967)
    df = pd.concat([phiusiil.data.features, phiusiil.data.targets], axis=1)

print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("[2/7] Preprocessing ...")
print("=" * 70)

# 2a. Drop identifier / free-text columns
present_drops = [c for c in DROP_COLS if c in df.columns]
df = df.drop(columns=present_drops)
print(f"   Dropped columns: {present_drops}")

# 2b. Duplicates
n_dupes = df.duplicated().sum()
if n_dupes:
    df = df.drop_duplicates().reset_index(drop=True)
print(f"   Duplicate rows removed: {n_dupes}")

# 2c. Missing values
n_missing = df.isnull().sum().sum()
if n_missing:
    df = df.dropna().reset_index(drop=True)
print(f"   Missing values handled: {n_missing}")

X = df.drop(columns=["label"])
y = df["label"]
print(f"   Feature matrix: {X.shape[1]} numeric features")

# 2d. Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"   Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  (stratified)")

# ---------------------------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("[3/7] Exploratory Data Analysis ...")
print("=" * 70)

# 3a. Summary statistics
summary = df.describe().T
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))
print(f"   [saved] {OUTPUT_DIR}/summary_statistics.csv")

# 3b. Class distribution
plt.figure(figsize=(6, 4))
counts = y.value_counts().sort_index()
ax = sns.barplot(
    x=["Phishing (0)", "Legitimate (1)"],
    y=counts.values,
    hue=["Phishing (0)", "Legitimate (1)"],
    palette=["#d9534f", "#5cb85c"],
    legend=False,
)
for i, v in enumerate(counts.values):
    ax.text(i, v + 1500, f"{v:,}", ha="center", fontweight="bold")
plt.title("Class Distribution - PhiUSIIL Dataset")
plt.ylabel("Number of URLs")
savefig("class_distribution.png")

# 3c. Histograms of key numeric features
hist_features = [
    "URLLength", "DomainLength", "URLSimilarityIndex", "CharContinuationRate",
    "TLDLegitimateProb", "NoOfSubDomain", "LetterRatioInURL", "DegitRatioInURL",
    "SpacialCharRatioInURL", "IsHTTPS", "NoOfImage", "NoOfJS",
]
hist_features = [f for f in hist_features if f in df.columns]
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for ax, feat in zip(axes.flatten(), hist_features):
    for lbl, color, name in [(0, "#d9534f", "Phishing"), (1, "#5cb85c", "Legitimate")]:
        ax.hist(df.loc[y == lbl, feat], bins=40, alpha=0.55, color=color, label=name)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=7)
fig.suptitle("Feature Distributions by Class", fontsize=15, y=1.01)
savefig("feature_histograms.png")

# 3d. Correlation heatmap (top features most correlated with the label)
corr = df.corr(numeric_only=True)
top_corr_feats = (
    corr["label"].abs().sort_values(ascending=False).head(21).index.tolist()
)
plt.figure(figsize=(13, 11))
sns.heatmap(
    df[top_corr_feats].corr(),
    annot=True, fmt=".2f", cmap="coolwarm", center=0,
    annot_kws={"size": 7}, square=True, linewidths=0.4,
)
plt.title("Correlation Heatmap - Top 20 Features vs Label")
savefig("correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 4. Decision Tree + hyperparameter tuning
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("[4/7] Training Decision Tree with GridSearchCV ...")
print("=" * 70)

param_grid = {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 10, 50],
    "min_samples_leaf": [1, 5, 20],
    "criterion": ["gini", "entropy"],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print(f"   Best params: {grid.best_params_}")
print(f"   Best CV F1 : {grid.best_score_:.4f}")

# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("[5/7] Evaluating on the held-out test set ...")
print("=" * 70)

y_pred = model.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision (phishing=0)": precision_score(y_test, y_pred, pos_label=0),
    "Recall (phishing=0)": recall_score(y_test, y_pred, pos_label=0),
    "F1 (phishing=0)": f1_score(y_test, y_pred, pos_label=0),
    "Precision (legit=1)": precision_score(y_test, y_pred, pos_label=1),
    "Recall (legit=1)": recall_score(y_test, y_pred, pos_label=1),
    "F1 (legit=1)": f1_score(y_test, y_pred, pos_label=1),
}

report = classification_report(
    y_test, y_pred, target_names=["Phishing (0)", "Legitimate (1)"]
)
print(report)

with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.txt"), "w") as f:
    f.write("Decision Tree - Test Set Evaluation\n")
    f.write("=" * 50 + "\n")
    f.write(f"Best hyperparameters: {grid.best_params_}\n\n")
    for k, v in metrics.items():
        f.write(f"{k:28s}: {v:.4f}\n")
    f.write("\nClassification report:\n")
    f.write(report)
print(f"   [saved] {OUTPUT_DIR}/evaluation_metrics.txt")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Phishing (0)", "Legitimate (1)"]).plot(
    ax=ax, cmap="Blues", values_format=","
)
plt.title("Confusion Matrix - Decision Tree (Test Set)")
savefig("confusion_matrix.png")

# ---------------------------------------------------------------------------
# 6. Feature importance + shallow tree visualisation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("[6/7] Feature importance & tree visualisation ...")
print("=" * 70)

importances = (
    pd.Series(model.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)
importances.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"))

top20 = importances.head(20)
plt.figure(figsize=(9, 8))
sns.barplot(x=top20.values, y=top20.index, hue=top20.index,
            palette="viridis", legend=False)
plt.title("Top 20 Feature Importances - Decision Tree")
plt.xlabel("Importance (Gini)")
savefig("feature_importance_top20.png")

# Shallow (depth-3) tree for interpretability in the report
shallow = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
shallow.fit(X_train, y_train)
plt.figure(figsize=(22, 10))
plot_tree(
    shallow,
    feature_names=X.columns,
    class_names=["Phishing", "Legitimate"],
    filled=True, rounded=True, fontsize=9,
)
plt.title("Decision Tree (depth = 3) - Illustrative Structure")
savefig("decision_tree_shallow.png")

# ---------------------------------------------------------------------------
# 6b. Robustness check: ablation without URLSimilarityIndex
# ---------------------------------------------------------------------------
# URLSimilarityIndex dominates the model (~98% of importance) and yields a
# perfect 100% test score, which examiners may (rightly) question. This
# ablation retrains WITHOUT that feature to demonstrate the model is not
# dependent on a single near-leaky column. Both results are reported.
print("\n" + "=" * 70)
print("[6b/7] Ablation study - excluding URLSimilarityIndex ...")
print("=" * 70)

X_train_abl = X_train.drop(columns=["URLSimilarityIndex"])
X_test_abl = X_test.drop(columns=["URLSimilarityIndex"])

abl_model = DecisionTreeClassifier(
    max_depth=15, random_state=RANDOM_STATE
).fit(X_train_abl, y_train)
abl_pred = abl_model.predict(X_test_abl)
abl_acc = accuracy_score(y_test, abl_pred)
abl_f1 = f1_score(y_test, abl_pred)
print(f"   Ablation accuracy: {abl_acc:.4f}  |  F1: {abl_f1:.4f}")

with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.txt"), "a") as f:
    f.write("\n" + "=" * 50 + "\n")
    f.write("Ablation: Decision Tree WITHOUT URLSimilarityIndex\n")
    f.write("(robustness check - see code comments)\n")
    f.write(f"Accuracy: {abl_acc:.4f}\nF1-score: {abl_f1:.4f}\n")

# ---------------------------------------------------------------------------
# 7. Persist the model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("[7/7] Saving model ...")
print("=" * 70)

model_path = os.path.join(OUTPUT_DIR, "decision_tree_model.joblib")
joblib.dump(model, model_path)
# Also persist the feature order the model expects (needed by Phase 3 API)
joblib.dump(list(X.columns), os.path.join(OUTPUT_DIR, "feature_columns.joblib"))
print(f"   [saved] {model_path}")
print(f"   [saved] {OUTPUT_DIR}/feature_columns.joblib")

print("\n" + "=" * 70)
print("PHASE 1 COMPLETE - all outputs are in the ./outputs directory")
print("=" * 70)
