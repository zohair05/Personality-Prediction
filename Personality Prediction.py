import pandas as pd
import os
import pickle
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# 1. Download NLTK resources 
nltk.download('punkt')
nltk.download('wordnet')

# 2. Preprocessing Function (Lemmatization)
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# 3. Load Dataset
data_path = "personality_project/dataset.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Make sure the CSV file exists.")
df = pd.read_csv(data_path)

# 4. Train / Retrain Model Function
def train_model(dataset):
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["text"], dataset["label"], test_size=0.3, random_state=42, stratify=dataset["label"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words='english', preprocessor=preprocess)),
        ("model", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\n📊 Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return pipeline

# 5. Train Initial Model
model = train_model(df)

# Cross-validation for stable accuracy
cv_scores = cross_val_score(model, df['text'], df['label'], cv=5)
print(f"\n🔸 Cross-validation Accuracy: {cv_scores.mean():.2f}")

# 6. Save / Load Model
os.makedirs("personality_project", exist_ok=True)
model_path = "personality_project/personality_pipeline.pkl"

def save_model_and_data(model, dataset):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    dataset.to_csv(data_path, index=False)
    print("✅ Model and dataset saved.")

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("\n📂 Loaded existing model.")

# 7. Prepare for Similarity Search
tfidf_vectorizer = model.named_steps["tfidf"]
tfidf_matrix = tfidf_vectorizer.transform(df["text"])

# 8. Interactive Loop
print("\n🔸 Personality Prediction System")

while True:
    user_text = input("\nTell us what you enjoy doing (or type 'exit' to stop): ").strip()
    if user_text.lower() == "exit":
        print("👋 Exiting program.")
        break

    # Vectorize user input
    user_vec = tfidf_vectorizer.transform([user_text])

    # Similarity check
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    max_sim_idx = np.argmax(sim_scores)
    max_sim = sim_scores[max_sim_idx]

    if max_sim > 0.6:  # lowered threshold slightly due to lemmatization
        predicted_label = df.loc[max_sim_idx, "label"]
        print(f"✅ Looks like you might be an: {predicted_label} (Similarity: {max_sim:.2f})")
        continue

    # Prediction + Confidence check
    proba = model.predict_proba([user_text])[0]
    max_prob = max(proba)
    pred_class = model.classes_[proba.argmax()]

    if max_prob < 0.6:
        print("\nI’m a bit unsure 🤔 ")
        while True:
            user_label = input("Please choose whether this post sounds introvert or extrovert.").strip().lower()
            if user_label in ["introvert", "extrovert"]:
                break
            print("Invalid input. Type 'introvert' or 'extrovert'.")

        # Add new example and retrain
        df = pd.concat([df, pd.DataFrame({"text": [user_text], "label": [user_label]})], ignore_index=True)
        model = train_model(df)
        tfidf_vectorizer = model.named_steps["tfidf"]
        tfidf_matrix = tfidf_vectorizer.transform(df["text"])
        save_model_and_data(model, df)
        print(f"✅ Model updated with new sentence labeled as {user_label}.")
    else:
        print(f"Predicted Personality: {pred_class} (Confidence: {max_prob:.2f})")
