from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.authRoutes import router as auth_router
from backend.routes.animeRoutes import router as anime_router
from backend.routes.userRoutes import router as user_router

app = FastAPI(title="YuGen API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(anime_router, tags=["anime"])
app.include_router(user_router, tags=["users"])

@app.get("/health")
def health():
    return {"status": "ok"}
