# YuGen — Anime Discovery Engine

YuGen is a full-stack anime discovery and recommendation platform. Browse thousands of titles, get intelligent personalised recommendations, and chat with an AI assistant that knows anime inside out.

---

## Table of Contents

- [How the App Works](#how-the-app-works)
- [Rating System](#rating-system)
- [Recommender System](#recommender-system)
- [Redis — What It Stores](#redis--what-it-stores)
- [Running with Docker (Quick Start)](#running-with-docker-quick-start)
- [CI/CD](#cicd)
- [DVC Pipeline](#dvc-pipeline)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)

---

## How the App Works

```
┌─────────────┐     HTTP      ┌─────────────────┐     SQL      ┌──────────────┐
│   Frontend  │ ────────────▶ │   FastAPI (API)  │ ──────────▶ │  PostgreSQL  │
│  React/Vite │ ◀──────────── │   Python 3.12    │             │  (user data) │
└─────────────┘               └────────┬────────┘             └──────────────┘
                                        │
                          ┌─────────────┼──────────────┐
                          ▼             ▼              ▼
                    ┌──────────┐  ┌──────────┐  ┌──────────┐
                    │  FAISS   │  │  Redis   │  │  Gemini  │
                    │  Index   │  │  Cache   │  │  + Tools │
                    │(artifacts│  │          │  │ (chatbot)│
                    │ folder)  │  └──────────┘  └──────────┘
                    └──────────┘
```

**User flows:**

- **Browse** — infinite scroll catalog of anime, filterable by title and genre. Data served from a preprocessed CSV baked into the API image.
- **Anime detail** — synopsis, stats (favorites, watching, completed), genres, studios, producers.
- **Rate** — logged-in users can rate any anime 1–10 directly from the detail page. Ratings are persisted immediately.
- **Recommendations** — logged-in users get hybrid personalised picks; guests get content-based similar anime.
- **Auth** — JWT-based register / login / logout. Tokens are revoked via Redis on logout.
- **Chatbot** — bottom-right AI assistant powered by Gemini with tool use (fuzzy search, FAISS similarity, web search via Tavily).

---

## Rating System

Logged-in users can rate any anime on a **1–10 scale** from the anime detail page. A single `POST /anime/{id}/rate` call triggers a four-step write:

```
POST /anime/{anime_id}/rate  { "rating": 8 }
         │
         ▼
1. DELETE + INSERT into user_anime_ratings   ← individual user score
         │
         ▼
2. COUNT + AVG from user_anime_ratings       ← recalculate aggregate stats
         │
         ▼
3. UPDATE ratings table                      ← num_ratings, mean_rating, popularity_score
         │
         ▼
4. Mirror both changes to in-memory DataFrames + rewrite both CSVs
```

### Why four steps?

| What | Why |
|---|---|
| `user_anime_ratings` table | Source of truth for individual scores; used by the hybrid recommender on retrain |
| `ratings` table | Aggregate stats (num_ratings, mean_rating, popularity_score) used for popularity ranking and the recommender reliability filter |
| In-memory DataFrames | `store.user_ratings_df` and `store.hybrid._ratings_df` are read on every request — updating them live means the **profile page and recommender see new ratings instantly** without a restart |
| CSV files | `artifacts/data_preprocessing/user_anime_ratings.csv` and `ratings.csv` are the inputs to the DVC pipeline — rewriting them means the **next `dvc repro` retrains on up-to-date data automatically** |

### popularity_score formula

Scores use a **Bayesian average** (same formula as IMDB) to prevent anime with very few ratings from unfairly dominating popularity rankings:

```
popularity_score = (v × R  +  m × C) / (v + m)

  v  =  num_ratings for this anime
  R  =  mean_rating for this anime
  m  =  50  (damping constant, reverse-engineered from dataset)
  C  =  Σ(v × R) / Σv  — weighted global mean across all anime
```

With a large `v`, the score converges to the actual mean. With a tiny `v`, it pulls toward the global average, so obscure anime can't rank above well-rated popular ones just from a handful of 10/10 votes.

### Endpoint

```
POST /anime/{anime_id}/rate
Authorization: Bearer <token>   ← required; 401 if missing or expired
Body: { "rating": <1–10> }
```

Returns `{ "message": "Rating saved successfully", "anime_id": ..., "rating": ... }`.

### Frontend

On the anime detail page, logged-in users see a **1–10 number bar** below the genre chips. The bar is pre-highlighted with their existing rating on page load (fetched in parallel with the anime detail). Hovering lights up buttons left-to-right; clicking submits immediately and shows a "Rating saved!" confirmation.

---

## Recommender System

### 1. Content-Based Recommender

Finds anime similar to a given title using TF-IDF + FAISS.

**How it works:**

1. A text corpus is built per anime from its genres, studios, producers, era, episode-count bin, and duration bin.
2. A TF-IDF vectorizer (up to 20 000 features, 1–2 ngrams) converts each corpus into a sparse vector.
3. Vectors are L2-normalised and stored in a **FAISS `IndexFlatIP`** (inner product = cosine similarity on normalised vectors).
4. At query time: look up the query anime's vector → `index.search(vec, k)` → return top-k most similar anime.

**Artifacts produced:**
```
artifacts/recommender/content_based/
    tfidf_vectorizer.pkl   ← fitted TF-IDF vectorizer
    faiss_index.bin        ← FAISS flat index
    anime_id_to_idx.json   ← maps anime_id → FAISS row index
```

**Params (`params.yaml`):**
```yaml
content_based:
  max_features: 20000   # vocabulary size
  ngram_range: [1, 2]   # unigrams + bigrams
  min_df: 2             # ignore terms appearing in < 2 anime
```

---

### 2. Hybrid Recommender (logged-in users)

Personalises recommendations based on a user's rating history.

**How it works:**

1. Load all ratings for the user from the preprocessed ratings CSV.
2. Build a **user taste vector** = weighted sum of content vectors:
   ```
   taste_vec = Σ (rating - mean_rating) × anime_vec
   ```
   Ratings above average contribute positively; below-average ratings subtract.
3. L2-normalise the taste vector.
4. Search the same FAISS index with the taste vector → returns anime that match the user's preferences, not a single title.

**Minimum ratings required:** 50 (configurable via `hybrid.min_num_ratings` in `params.yaml`). Users below this threshold fall back to content-based.

**No separate FAISS index needed** — the hybrid reuses the content-based index. At startup, the content-based recommender is loaded once and injected into the hybrid:
```python
self.hybrid._content = self.cb   # shared in-memory, no double loading
```

---

### 3. Evaluation

Both recommenders are evaluated using **genre overlap @ K**:

- For content-based: pick a random sample of anime, get top-K recommendations, measure what fraction share at least one genre with the seed.
- For hybrid: sample users with ≥ 20 ratings, build their taste vector from ratings ≥ 7.0 ("liked"), get top-K recs, check genre overlap.

Results are saved to `artifacts/evaluation/`.

---

## Redis — What It Stores

Redis serves two purposes in YuGen. Both use the same Redis instance (redis-stack).

### 1. JWT Token Blacklist

When a user logs out, their JWT token is **revoked** by storing it in Redis until it naturally expires.

```
Key:   revoked:<full_jwt_token>
Value: "1"
TTL:   remaining seconds until the token's exp timestamp
```

On every authenticated request, the middleware checks `EXISTS revoked:<token>`. If the key exists, the request is rejected with 401. TTL ensures Redis auto-cleans expired tokens — no manual cleanup needed.

### 2. Chat History

Every chatbot message is persisted in Redis as an append-only list per session thread.

```
Key:    chat:history:<thread_id>
Value:  Redis list of JSON strings, e.g.:
        [
          {"role": "human",  "content": "recommend something like Death Note"},
          {"role": "ai",     "content": "You'd enjoy..."}
        ]
```

This is a secondary store — the primary conversation memory lives inside LangGraph's checkpointer (also Redis-backed via `RedisSaver`). The `chat:history:*` keys are used by the `/chat/{thread_id}/history` API endpoint to return history to the frontend.

**LangGraph checkpointer keys** (managed internally by LangGraph, not by app code):
```
langgraph:checkpoint:<thread_id>:*
```

### Fallback behaviour

If Redis is unreachable at startup, the app falls back to an **in-process `MemorySaver`** for LangGraph (history lost on restart) and raises errors on logout (token blacklist requires Redis). The anime catalog and recommendations continue working normally.

---

## Running with Docker (Quick Start)

### Prerequisites
- Docker Desktop
- A Google Gemini API key ([get one free](https://aistudio.google.com/))
- A Tavily API key ([get one free](https://tavily.com/))

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/shreyaan16/yugen
cd yugen

# 2. Create your .env
cp .env.example .env
# Edit .env — fill in at minimum:
#   GOOGLE_API_KEY=...
#   TAVILY_API_KEY=...
#   JWT_SECRET=any-random-string
#   DB_PASSWORD=your-chosen-password

# 3. Start everything
docker-compose up
```

| Service   | URL                        |
|-----------|----------------------------|
| Frontend  | http://localhost:5173       |
| API       | http://localhost:8000       |
| API docs  | http://localhost:8000/docs  |

Docker pulls `shreyaan16/yugen-api:latest` and `shreyaan16/yugen-frontend:latest` from Docker Hub. Anime catalog and recommendation artifacts are **baked into the image** — no extra downloads needed.

The postgres and redis containers start automatically. On first run, tables are created automatically via SQLAlchemy.

---

## CI/CD

Every push to `main` triggers a GitHub Actions workflow (`.github/workflows/docker-publish.yml`) that builds and pushes both Docker images to Docker Hub in **parallel**.

```
push to main
      │
      ├─── job: backend  ──▶  docker build .               ──▶  shreyaan16/yugen-api:latest
      │                                                          shreyaan16/yugen-api:<sha>
      │
      └─── job: frontend ──▶  docker build ./frontend      ──▶  shreyaan16/yugen-frontend:latest
                                                                 shreyaan16/yugen-frontend:<sha>
```

Each image is tagged with both `:latest` (for easy `docker pull`) and the full commit SHA (for precise rollbacks).  
Layer caching (`type=gha`) is enabled — unchanged layers (pip installs, npm installs) are reused across runs.

### Required GitHub Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token *(Account Settings → Security → New Access Token)* |
| `VITE_API_BASE_URL` | Public URL of your API, e.g. `http://your-server:8000` *(leave empty to keep the `localhost:8000` default)* |

---

## DVC Pipeline

DVC manages the ML training pipeline. It tracks which stages need re-running based on file and parameter changes — skipping expensive stages when nothing has changed.

### Pipeline stages

```
loading → data_ingestion → data_preprocessing → recommender_train → recommender_eval
                                              ↘ chatbot (parallel)
```

| Stage | What it does |
|---|---|
| `loading` | Exports PostgreSQL tables → Azure Blob Storage. Smart: skips tables unchanged since last run (checked via `pg_stat_user_tables`). |
| `data_ingestion` | Downloads CSVs from Azure → `artifacts/data_ingestion/` |
| `data_preprocessing` | Cleans, bins, and aggregates → `artifacts/data_preprocessing/` |
| `recommender_train` | Trains TF-IDF + FAISS content-based recommender → `artifacts/recommender/` |
| `chatbot` | Runs chatbot demo (no output artifact — always re-runs) |
| `recommender_eval` | Evaluates both recommenders → `artifacts/evaluation/` |

---

### Running DVC

#### Option A — Retrain with fresh data from your database

Use this when your PostgreSQL database has been updated with new anime or ratings and you want to retrain the models.

```bash
# Step 1: Force the loading stage to re-export from PostgreSQL → Azure
dvc repro --force loading

# Step 2: Let DVC cascade — ingestion → preprocessing → training → eval
dvc repro
```
If the manifest file hash changes, all downstream stages automatically re-run.

#### Option B — Retrain with changed hyperparameters only

Use this when you've edited `params.yaml` (e.g. tuning `max_features` or `liked_threshold`) but the data hasn't changed.

```bash
dvc repro
```

DVC detects the param change and re-runs only the affected stages:
- `recommender_train` (if `content_based` or `hybrid` params changed)
- `recommender_eval` (if `evaluation` params changed)
- `chatbot` (if `chatbot` params changed)

Loading, ingestion, and preprocessing are **skipped** — no database or Azure access.

#### Option C — Force a specific stage

```bash
dvc repro --force <stage_name>
# e.g.
dvc repro --force recommender_train
```

#### After retraining — rebuild and push the Docker image

Commit the updated artifacts and push to `main` — the CI/CD pipeline will rebuild and push the Docker image automatically.

```bash
git add artifacts/
git commit -m "retrain: updated recommender artifacts"
git push origin main
```

Or rebuild manually:

```bash
docker-compose build api
docker push shreyaan16/yugen-api:latest
```

---

## Project Structure

```
YuGen/
├── frontend/               # React + Vite frontend
│   ├── src/
│   │   ├── pages/          # Home, AnimeDetail, Profile
│   │   ├── components/     # AnimeCard, Chatbot, AuthPanel, ...
│   │   └── api.js          # all fetch calls to the backend
│   └── Dockerfile
│
├── backend/                # FastAPI backend
│   ├── routes/             # authRoutes, animeRoutes, userRoutes
│   ├── controllers/        # business logic
│   ├── services/
│   │   ├── store.py        # app singleton (loads CSVs + models once)
│   │   ├── redis_client.py # Redis connection
│   │   ├── token_blacklist.py  # JWT revocation via Redis
│   │   └── chat_history.py    # chat persistence via Redis
│   └── main.py
│
├── recommender/            # ML recommender package
│   ├── components/
│   │   ├── content_based.py   # TF-IDF + FAISS
│   │   ├── hybrid.py          # user taste vector + FAISS
│   │   └── evaluation.py      # genre overlap @ K
│   └── pipeline/
│
├── chatbot/                # LangGraph AI assistant
│   ├── agent/              # AnimeAgent (Gemini + tools)
│   └── tools/              # fuzzy search, FAISS similarity, web search
│
├── ingestion/              # Data ingestion package
│   ├── components/
│   │   ├── loading.py         # PostgreSQL → Azure (smart, stat-based)
│   │   ├── data_ingestion.py  # Azure → local CSV
│   │   └── data_preprocessing.py
│   └── pipeline/
│
├── artifacts/              # Generated by DVC pipeline (baked into Docker image)
│   ├── data_ingestion/
│   ├── data_preprocessing/
│   └── recommender/
│       └── content_based/
│
├── dvc.yaml                # Pipeline definition
├── params.yaml             # Hyperparameters
├── docker-compose.yml
├── Dockerfile              # Backend image
└── .env.example
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values.

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ | Gemini API key for the chatbot |
| `TAVILY_API_KEY` | ✅ | Web search API key for the chatbot |
| `JWT_SECRET` | ✅ | Any random string for signing JWTs |
| `DB_PASSWORD` | ✅ | Password for the docker-compose postgres container |
| `DATABASE_URL` | Local dev only | Postgres URL (overridden in docker-compose) |
| `REDIS_HOST` | ✅ | `redis` in Docker, `localhost` for local dev |
| `CONNECTION_STRING` | Pipeline only | Azure Blob Storage connection string (only needed to run `dvc repro`) |
| `GOOGLE_API_KEY` | ✅ | Also used by the loading pipeline for Azure auth |
| `JWT_EXPIRE_MINUTES` | Optional | Token lifetime, default 1800 (30 min) |
