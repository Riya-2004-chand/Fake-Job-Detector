from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import json
from database import SessionLocal, JobResult

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class JobInput(BaseModel):
    text: str

def extract_flags(text: str):
    flags = []
    t = text.lower()
    if re.search(r"gmail|yahoo|hotmail", t):
        flags.append("Personal email contact detected")
    if re.search(r"urgent|apply now|limited seats|hurry", t):
        flags.append("Urgency language detected")
    if re.search(r"registration fee|pay to join|deposit required", t):
        flags.append("Asks for money upfront")
    if re.search(r"no experience required|anyone can apply", t):
        flags.append("Suspiciously low requirements")
    if len(re.findall(r"!", text)) > 5:
        flags.append("Excessive use of exclamation marks")
    if re.search(r"work from home|earn \d+k|per day earning", t):
        flags.append("Suspicious earning claims")
    return flags

@app.post("/predict")
def predict(job: JobInput):
    vec = vectorizer.transform([job.text])
    prob = float(model.predict_proba(vec)[0][1])
    flags = extract_flags(job.text)
    if prob > 0.7:
        risk = "High"
    elif prob > 0.4:
        risk = "Medium"
    else:
        risk = "Low"
    prediction = "FAKE" if prob > 0.45 or len(flags) >= 2 else "REAL"
    db = SessionLocal()
    record = JobResult(
        text=job.text[:500],
        prediction=prediction,
        confidence=round(prob * 100, 2),
        risk_level=risk,
        flags=json.dumps(flags)
    )
    db.add(record)
    db.commit()
    db.close()
    return {
        "prediction": prediction,
        "confidence": round(prob * 100, 2),
        "risk_level": risk,
        "flags": flags
    }

@app.get("/history")
def history():
    db = SessionLocal()
    records = db.query(JobResult).order_by(JobResult.created_at.desc()).limit(20).all()
    db.close()
    return [
        {
            "id": r.id,
            "text": r.text[:100],
            "prediction": r.prediction,
            "confidence": r.confidence,
            "risk_level": r.risk_level,
            "flags": json.loads(r.flags),
            "created_at": str(r.created_at)
        }
        for r in records
    ]
