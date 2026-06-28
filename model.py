import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

print('Starting...', flush=True)
df = pd.read_csv('fake_job_postings.csv')
print(f'Loaded {len(df)} rows', flush=True)
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
df[text_cols] = df[text_cols].fillna('')
df['combined_text'] = df[text_cols].apply(lambda x: ' '.join(x), axis=1)
X = df['combined_text']
y = df['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Vectorizing...', flush=True)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print('Training model...', flush=True)
model = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
model.fit(X_train_vec, y_train)
print('Evaluating...', flush=True)
print(classification_report(y_test, model.predict(X_test_vec)))
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print('Model saved.', flush=True)
