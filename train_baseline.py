import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os

# 1. Load the clean training data
print("Loading cleaned training data...")
df = pd.read_csv("data/processed/labeled_training_data.csv")

# Check for any lingering missing values in the message column
df = df.dropna(subset=['message', 'sentiment'])

# 2. Separate features (X) and labels (y)
X = df['message']
y = df['sentiment']

# 3. Split into Train and Test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorize the text (Convert words into numbers)
# We limit to 5000 features to keep the model fast and prevent overfitting
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train the Baseline Model (Logistic Regression)
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# 6. Evaluate the Model
print("\n--- Evaluation Results ---")
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Weighted F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/baseline_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
print("\nBaseline model and vectorizer saved to models/ folder!")