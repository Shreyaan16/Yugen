import sys, os, logging
# Ensure the project root (parent of backend/) is on the path so that both
# the `backend` package and sibling packages (recommender, chatbot, ingestion)
# are importable regardless of which directory uvicorn is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.authRoutes import router as auth_router
from backend.routes.animeRoutes import router as anime_router
from backend.routes.userRoutes import router as user_router
from backend.database import Base, engine

log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables on startup — fails gracefully if DB is unreachable
    try:
        Base.metadata.create_all(bind=engine)
        log.info("[startup] Database tables created / verified.")
    except Exception as exc:
        log.warning("[startup] Could not create DB tables: %s", exc)
    yield

app = FastAPI(title="YuGen API", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(anime_router, tags=["anime"])
app.include_router(user_router, tags=["users"])

@app.get("/health")
def health():
    return {"status": "ok"}
