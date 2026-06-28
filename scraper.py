import requests
from bs4 import BeautifulSoup
import json

def scrape_remoteok():
    url = "https://remoteok.com/api"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    jobs = res.json()
    results = []
    for job in jobs[1:11]:  # first 10 jobs
        text = f"{job.get('position','')} {job.get('company','')} {job.get('description','')}"
        results.append({
            "title": job.get("position", ""),
            "company": job.get("company", ""),
            "text": text
        })
    return results

def analyze(text):
    res = requests.post("http://127.0.0.1:8000/predict",
        json={"text": text},
        headers={"Content-Type": "application/json"})
    return res.json()

if __name__ == "__main__":
    print("Scraping jobs...")
    jobs = scrape_remoteok()
    print(f"Found {len(jobs)} jobs\n")
    for job in jobs:
        result = analyze(job["text"])
        print(f"Title: {job['title']}")
        print(f"Company: {job['company']}")
        print(f"Prediction: {result['prediction']} | Confidence: {result['confidence']}% | Risk: {result['risk_level']}")
        if result['flags']:
            print(f"Flags: {', '.join(result['flags'])}")
        print("-" * 50)
