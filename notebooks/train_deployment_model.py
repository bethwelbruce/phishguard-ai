"""
Deployment Model Trainer - PhishGuard AI
=========================================
The Phase 1 model uses all 50 PhiUSIIL features, many of which (LineOfCode,
NoOfImage, HasSocialNet, ...) require fetching and parsing the target web
page. A live "type a URL, get an answer" app cannot compute those honestly
in real time.

This script trains a SECOND Decision Tree using ONLY features that can be
derived from the URL string itself, so the web app's predictions are
truthful and instantaneous. It also saves a background data sample needed
by the LIME explainer at inference time.

Run from the repo root:
    python notebooks/train_deployment_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
CSV = "PhiUSIIL_Phishing_URL_Dataset.csv"
OUT = "outputs"

# Features computable from the URL string alone (no page fetch required).
# TLDLegitimateProb is included via a lookup table exported below.
URL_ONLY_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLegitimateProb",
    "URLCharProb", "TLDLength", "NoOfSubDomain", "HasObfuscation",
    "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL",
    "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]

if not os.path.exists(CSV):
    raise SystemExit(f"Dataset not found: {CSV} - place it in the repo root.")

df = pd.read_csv(CSV, encoding="utf-8").drop_duplicates()

# Export TLD -> legitimate probability lookup for the live feature extractor
tld_lookup = (
    df.groupby(df["TLD"].str.lower())["TLDLegitimateProb"].median().to_dict()
)
joblib.dump(tld_lookup, os.path.join(OUT, "tld_probability_lookup.joblib"))

X = df[URL_ONLY_FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model = DecisionTreeClassifier(
    max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)
print(f"Deployment model (URL-only features) - Accuracy: {acc:.4f} | F1: {f1:.4f}")
print(classification_report(y_test, pred, target_names=["Phishing", "Legitimate"]))

joblib.dump(model, os.path.join(OUT, "deployment_model.joblib"))
joblib.dump(URL_ONLY_FEATURES, os.path.join(OUT, "deployment_features.joblib"))

# Background sample for the LIME tabular explainer (kept small for speed)
bg = X_train.sample(1000, random_state=RANDOM_STATE).to_numpy()
np.save(os.path.join(OUT, "lime_background.npy"), bg)

# Persist test metrics for the /metrics endpoint
metrics = {
    "accuracy": round(float(acc), 4),
    "f1_score": round(float(f1), 4),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": len(URL_ONLY_FEATURES),
    "model": "DecisionTreeClassifier(max_depth=12, min_samples_leaf=5)",
}
joblib.dump(metrics, os.path.join(OUT, "deployment_metrics.joblib"))
print("Saved: deployment_model, deployment_features, lime_background,",
      "tld_probability_lookup, deployment_metrics -> ./outputs")
