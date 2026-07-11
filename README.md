# Fake Job Detector 🔍

An ML-powered web app that detects fraudulent job postings to protect fresher students from scams.

## 🚀 Live Demo
**https://fake-job-frontend-omega.vercel.app**

## 🛠️ Tech Stack
- **Frontend:** Next.js, Tailwind CSS
- **Backend:** FastAPI, Python
- **ML Model:** Logistic Regression + TF-IDF Vectorizer
- **Database:** SQLite + SQLAlchemy
- **Deployment:** Vercel (frontend) + Render (backend)

## 📊 How It Works
1. User pastes a job description
2. TF-IDF vectorizer converts text to features
3. Logistic Regression model predicts fake/real
4. Keyword flags detect red flags (money upfront, urgency, etc.)
5. Result shown with confidence score and risk level

## 🤖 ML Model
- Dataset: 17,880 job postings (Kaggle)
- Algorithm: Logistic Regression with class_weight='balanced'
- Recall for fake jobs: **87%**

## 📁 Project Structure
- `api.py` — FastAPI backend with /predict and /history endpoints
- `model.py` — ML model training script
- `database.py` — SQLite database setup
- `scraper.py` — Job scraper from RemoteOK

## 👩‍💻 Developer
Riya Chand — B.Tech IT, Heritage Institute of Technology, Kolkata
