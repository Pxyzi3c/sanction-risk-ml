from fastapi import FastAPI
from routes.predict import router as predict_router
from routes.match import router as matches_router

app = FastAPI(
    title="Sanction Match Predictor API",
    version="1.0.0",
)

# -----------------------------
# Routes
# -----------------------------
app.include_router(predict_router, prefix="/predict_match")
app.include_router(matches_router, prefix="/matches")