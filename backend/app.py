
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from backend.model import predict_review  # Import prediction function from model.py

# -------------------------
# Initialize FastAPI
# -------------------------
app = FastAPI(title="Medicine Review Sentiment API")

# Allow CORS for testing (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request Model
# -------------------------
class Review(BaseModel):
    text: str

# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {"message": "Medicine Review Sentiment API is running"}

# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict")
def predict(review: Review):
    try:
        predictions = predict_review(review.text)
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# Serve Frontend (Static Files)
# -------------------------
# Mount static frontend AFTER API routes

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
