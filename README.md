# PhishGuard AI 🛡️

AI-powered phishing URL detection with a Decision Tree classifier and
explainable AI (SHAP + LIME), trained on the **PhiUSIIL Phishing URL
Dataset** (UCI ML Repository, ID 967 — 235,795 URLs, 54 features).

**Stack:** Python (FastAPI) backend · React (Vite + Chart.js) frontend ·
GitHub Actions CI/CD.

---

## Repository structure

```
phishguard-ai/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app - all API endpoints
│   │   └── features.py        # URL -> feature-vector extractor
│   ├── tests/test_api.py      # pytest suite (runs in CI)
│   └── requirements.txt
├── frontend/                  # React app (Vite)
│   └── src/
│       ├── App.jsx            # Tabs: Scanner / Dashboard / History
│       ├── api.js             # API client
│       └── components/        # Scanner, Dashboard, History
├── notebooks/
│   ├── phase1_train.py            # Phase 1: EDA + tuned model (report charts)
│   └── train_deployment_model.py  # URL-only model used by the live app
├── outputs/                   # Trained models, plots, metrics
└── .github/workflows/ci.yml   # CI/CD pipeline
```

## Two models — why?

| Model | Features | Test accuracy | Purpose |
|---|---|---|---|
| Phase 1 (`decision_tree_model.joblib`) | All 50 numeric features | 100.00%* | Report charts, EDA, evaluation |
| Deployment (`deployment_model.joblib`) | 20 URL-derivable features | 99.58% | Live API predictions |

Many PhiUSIIL features (`LineOfCode`, `NoOfImage`, `HasSocialNet`, …)
require fetching the target web page, which a live scanner cannot do
honestly in real time. The deployment model therefore uses only features
computable from the URL string itself.

\* The perfect Phase 1 score is driven by `URLSimilarityIndex`
(~98.6% of importance). An ablation retrained without it scores 99.88% —
see `outputs/evaluation_metrics.txt` and comments in
`notebooks/phase1_train.py`.

## Quick start

### 0. Dataset (for retraining only)

The dataset CSV (~57 MB) is **not** committed. To retrain, download it
from the [UCI repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
and place `PhiUSIIL_Phishing_URL_Dataset.csv` in the repo root, then:

```bash
python notebooks/phase1_train.py            # Phase 1 outputs
python notebooks/train_deployment_model.py  # live-app model
```

The pretrained artifacts in `outputs/` are committed, so the app runs
without the dataset.

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — the dev server proxies `/api` to the
backend on port 8000.

### 3. Tests

```bash
cd backend && pytest tests/ -v
```

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Liveness check |
| POST | `/api/predict` | `{"url": "..."}` → prediction, confidence, features |
| POST | `/api/predict/batch` | `{"urls": [...]}` → up to 100 predictions |
| GET | `/api/metrics` | Deployment-model test metrics |
| GET | `/api/feature-importance` | Sorted feature importances |
| POST | `/api/explain` | `{"url": "..."}` → SHAP values + LIME rules |

**Label convention (PhiUSIIL):** `1 = legitimate`, `0 = phishing`.

## CI/CD

Every push and pull request to `main` triggers GitHub Actions
(`.github/workflows/ci.yml`):

1. **Backend job** — installs dependencies, syntax-checks all Python,
   runs the pytest suite against the real committed model artifacts.
2. **Frontend job** — `npm ci` + production build; uploads the `dist/`
   bundle as a build artifact.
3. **Deploy gate** — runs only on pushes to `main` after both jobs pass.
   A commented Render/Vercel hook is included; add the platform secret
   and uncomment to enable one-click continuous deployment.

## Dataset citation

Prasad, A. & Chandra, S. (2023). *PhiUSIIL: A diverse security profile
empowered phishing URL detection framework based on similarity index and
incremental learning.* Computers & Security. UCI ML Repository ID 967.

## Disclaimer

Academic project. Predictions are probabilistic and should not be the
sole basis for security decisions.
