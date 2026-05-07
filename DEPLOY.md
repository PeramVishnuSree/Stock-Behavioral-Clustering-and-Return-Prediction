# Deployment Instructions — Streamlit Community Cloud

The repo is ready: code + precomputed `data_cache/` parquet artifacts are pushed to GitHub. You just need to connect it to Streamlit Cloud (free tier, ~5 minutes).

## One-time Streamlit Cloud setup

1. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with the same GitHub account that owns the repo (`PeramVishnuSree`). You're already authorized for Streamlit Cloud since you have `research-papers.streamlit.app` deployed.

2. Click **"New app"** in the top right corner.

3. Fill out the deployment form:
   - **Repository:** `PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick a custom subdomain — suggestion: `stock-clustering` → `https://stock-clustering.streamlit.app`

4. Click **Deploy**.

## What happens next

- Streamlit Cloud clones the repo, reads `requirements.txt`, and installs all dependencies (~2-3 min for first build).
- Once running, the dashboard reads from the precomputed `data_cache/` parquet files — instant load, no pipeline runs on the cloud.
- Your live URL appears in the dashboard manager — that's the URL to share.

## Repo URL

`https://github.com/PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction`

## Estimated final URL

`https://<your-chosen-subdomain>.streamlit.app`

## Build time

- First deploy: 2-3 minutes (one-time pip install)
- Cold restarts: 10-30 seconds (cached dependencies)

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails on `requirements.txt` | Version conflict | Re-pin a working version locally and push |
| Pages show "artifacts not found" | data_cache missing | Re-push `data_cache/` directory to GitHub |
| Dashboard times out on first load | Memory limit (free tier) | Open one page at a time; the parquet files are within the 1 GB limit |

## Local development

After cloning, anyone can reproduce the full project:

```bash
git clone https://github.com/PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction.git
cd Stock-Behavioral-Clustering-and-Return-Prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: re-run pipeline (if you want to refresh data; takes 5-10 min)
python pipeline.py

# Or just launch the dashboard with the committed cache
streamlit run app.py
```

Generated outputs:
- `REPORT.md` — detailed implementation report
- `PRESENTATION.pptx` — 25-slide deck
- `QA.md` — anticipated Q&A
