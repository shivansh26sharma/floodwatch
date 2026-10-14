# Deploying the dashboard

Goal: a public URL your professor and team can open in a browser, with no
installation. Free, about 20 minutes.

**Streamlit Community Cloud** is the right host for this — it is free, deploys
straight from a GitHub repo, and auto-redeploys on every push.

---

## Why this repo is already deploy-ready

Three problems normally break a first deploy. All three are handled:

| Problem | Fix in this repo |
|---|---|
| The app needs `data/processed/*.parquet` and trained models, which are gitignored build artifacts — a fresh clone has none | `src/pipeline.py::ensure_artifacts()` runs on startup and rebuilds everything from the committed fixture (~20s, cached per server) |
| `torch` is ~800MB and blows the free tier's resource limit | Split requirements: `requirements.txt` is the slim deploy set (no torch, no statsmodels, no xgboost); `requirements-dev.txt` is the full local set |
| Streamlit's file watcher restarts the app mid-build when the bootstrap writes files | `.streamlit/config.toml` sets `fileWatcherType = "none"` |

The fixture is committed on purpose — it is the seed that makes a fresh clone
self-healing. That is also why `.gitignore` has `!data/fixtures/`.

---

## Steps

### 1. Push to GitHub

```bash
cd floodwatch
git init
git add .
git commit -m "FloodWatch: flood risk pipeline + dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/floodwatch.git
git push -u origin main
```

The repo must be **public** for the free tier (private repos are limited).

Before pushing, check nothing large or licence-restricted slipped in:

```bash
git ls-files | xargs du -ch 2>/dev/null | tail -1   # should be a few MB
```

### 2. Deploy

1. Go to **share.streamlit.io** and sign in with GitHub
2. **New app** → pick your repo, branch `main`
3. **Main file path:** `dashboard/app.py`
4. **Deploy**

First build takes 3–5 minutes (installing packages), then the bootstrap runs
for ~20 seconds on first page load. After that it is instant.

### 3. Share the URL

You get `https://<your-app-name>.streamlit.app`. That link goes in your
report, your slides, and your GitHub README.

---

## Known limits of the free tier

- **The app sleeps after ~7 days of inactivity.** It wakes on the next visit,
  but takes ~30s. **Open it an hour before your viva** so it is warm.
- ~1GB memory. Fine for this pipeline; it is why torch is excluded.
- Public by default. Anyone with the link can view it — which is what you
  want here, but do not put anything private in the repo.

---

## Before the demo

- [ ] Open the URL an hour early so the app is awake
- [ ] Check the fixture banner says what you expect (it should be visible
      until you switch to real data — that honesty is deliberate)
- [ ] Have a screen recording as backup. Campus wifi fails during vivas.

---

## When you switch to real data

The deployed app rebuilds from whatever `config.yaml` points at. Two options:

**Option A — keep the public demo on the fixture.** Safest. The banner stays
up, nobody is misled, and you show real results from your report separately.

**Option B — deploy with real data.** Commit the *processed* parquet (usually
a few MB) and set `data.source: real`. Before you do:

> **Check the licence on every dataset you commit.** IMD, WRIS and DFO each
> have their own terms on redistribution. Putting government data in a public
> GitHub repo is republishing it. Cite the source and check you are allowed —
> "I only used it for a project" is not a licence.

If in doubt, deploy Option A and keep real data local.

---

## Alternatives if Streamlit Cloud doesn't work out

- **Hugging Face Spaces** — free, supports Streamlit, more generous memory
  (16GB on free CPU). Needs an `app_file:` entry in the Space README header.
- **Render / Railway** — free tiers are tighter than they used to be; only
  worth it if you need something Streamlit Cloud can't do.
