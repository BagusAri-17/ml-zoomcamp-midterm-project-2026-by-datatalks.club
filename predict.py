import os
import pickle
from typing import Any, Dict
from fastapi import FastAPI, HTTPException

# Initialize FastAPI app
app = FastAPI(
    title="Student Dropout Early Warning System API",
    description="Machine learning inference service predicting university student dropout risk using XGBoost.",
    version="1.0.0"
)

# Load trained DictVectorizer and XGBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.bin')

try:
    with open(MODEL_PATH, 'rb') as f_in:
        dv, model = pickle.load(f_in)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {str(e)}")


def categorize_risk(prob: float):
    """Categorize risk level and provide institutional recommendation based on probability."""
    if prob >= 0.70:
        return (
            "High Risk",
            "Urgent intervention recommended. Schedule academic tutoring, financial aid consultation, and faculty advisor check-in."
        )
    elif prob >= 0.40:
        return (
            "Moderate Risk",
            "Preventive monitoring recommended. Suggest course review and ensure tuition payment schedule is on track."
        )
    else:
        return (
            "Low Risk",
            "Student is in good academic standing. Continue regular academic progress tracking."
        )


@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint to verify service availability."""
    return {
        "status": "healthy",
        "service": "student-dropout-prediction-api",
        "version": "1.0.0"
    }


@app.post("/predict")
def predict(student: Dict[str, Any]):
    """Predict dropout probability and risk level for a single student."""
    if not student:
        raise HTTPException(status_code=400, detail="Student payload cannot be empty.")

    try:
        # Transform student feature dict using DictVectorizer
        X = dv.transform([student])

        # Compute dropout probability (class 1)
        prob = float(model.predict_proba(X)[0, 1])
        prediction = bool(prob >= 0.5)
        risk_level, recommendation = categorize_risk(prob)

        return {
            "dropout_probability": round(prob, 4),
            "dropout_prediction": prediction,
            "risk_level": risk_level,
            "recommendation": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=True)