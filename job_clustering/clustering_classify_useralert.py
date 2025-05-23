# main_pipeline.py

import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ==== USER CONFIGURATION ====
USER_PREFERRED_CLUSTERS = [0, 2]  # Update this based on your interest

# ==== LOAD LATEST SCRAPED JOB DATA ====
df = pd.read_csv("data/karkidi.csv")

# ==== CLEAN & PREPROCESS ====
df["Cleaned_Skills"] = df["Skills"].fillna("").str.lower().str.replace(",", " ")

# ==== CHECK IF MODEL EXISTS ====
model_path = "models"
vectorizer_path = os.path.join(model_path, "tfidf_vectorizer.pkl")
kmeans_path = os.path.join(model_path, "kmeans_model.pkl")
model_exists = os.path.exists(vectorizer_path) and os.path.exists(kmeans_path)

# Ensure model and data folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

if not model_exists:
    print("📊 No existing model found. Training new clustering model...")

    # === Train TF-IDF + KMeans ===
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["Cleaned_Skills"])

    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    df["Cluster"] = kmeans.fit_predict(X)

    # === Save Model ===
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(kmeans, kmeans_path)

    # === Save clustered data baseline ===
    df.to_csv("data/karkidi_clustered.csv", index=False)
    print("✅ Model trained and initial clustering complete.")
else:
    print("📁 Using saved model to classify new jobs...")

    # === Load Model ===
    vectorizer = joblib.load(vectorizer_path)
    kmeans = joblib.load(kmeans_path)

    # === Predict Clusters ===
    X_new = vectorizer.transform(df["Cleaned_Skills"])
    df["Cluster"] = kmeans.predict(X_new)

    # === Save new clustered jobs ===
    df.to_csv("data/karkidi_clustered_new.csv", index=False)

    # === Compare with previous data ===
    clustered_path = "data/karkidi_clustered.csv"
    if os.path.exists(clustered_path):
        df_old = pd.read_csv(clustered_path)
        old_jobs = set(df_old["Title"] + df_old["Company"])
        df["Identifier"] = df["Title"] + df["Company"]
        new_jobs = df[~df["Identifier"].isin(old_jobs)]

        # === Filter based on user preferred clusters ===
        matching_jobs = new_jobs[new_jobs["Cluster"].isin(USER_PREFERRED_CLUSTERS)]

        print("🔍 Checking for matching jobs in your preferred categories...")
        if matching_jobs.empty:
            print("🔕 No new jobs found in your preferred categories.")
        else:
            print(f"🔔 Found {len(matching_jobs)} new job(s) matching your preferences:\n")
            for _, row in matching_jobs.iterrows():
                print(f"🔹 {row['Title']} at {row['Company']}")
                print(f"📍 {row['Location']} | 🧠 {row['Skills']}\n")

        # === Update baseline with latest classified jobs
        df.drop(columns=["Identifier"], inplace=True, errors="ignore")
        df.to_csv("data/karkidi_clustered.csv", index=False)
    else:
        print("⚠️ No previous clustered data found. Creating initial baseline...")
        df.to_csv("data/karkidi_clustered.csv", index=False)
