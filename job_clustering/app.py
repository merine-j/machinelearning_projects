import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Karkidi Job Alerts", layout="wide")
st.title("🚀 Karkidi Job Monitor")

# === Load Data and Model ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("karkidi_clustered_new.csv")
    except:
        df = pd.DataFrame()
    return df

@st.cache_resource
def load_model():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    return vectorizer, kmeans

df = load_data()
vectorizer, kmeans = load_model()

# === Sidebar: User Preferences ===
st.sidebar.header("🔧 User Preferences")
available_clusters = list(range(kmeans.n_clusters))
preferred_clusters = st.sidebar.multiselect("Select preferred clusters", available_clusters, default=[0, 1])

# === Alert Matching Jobs ===
if df.empty:
    st.warning("⚠️ No clustered data found. Please run the backend script.")
else:
    st.success(f"✅ Loaded {len(df)} jobs")
    
    new_jobs_df = df.copy()
    
    if "Title" in df.columns and "Company" in df.columns:
        # Compare with old job identifiers
        if os.path.exists("karkidi_clustered.csv"):
            old_df = pd.read_csv("karkidi_clustered.csv")
            old_ids = set(old_df["Title"] + old_df["Company"])
            new_jobs_df["Identifier"] = new_jobs_df["Title"] + new_jobs_df["Company"]
            new_jobs_df = new_jobs_df[~new_jobs_df["Identifier"].isin(old_ids)]

        matches = new_jobs_df[new_jobs_df["Cluster"].isin(preferred_clusters)]

        st.subheader("🔔 New Jobs Matching Your Preferred Categories")

        if matches.empty:
            st.info("No new matching jobs found.")
        else:
            for _, row in matches.iterrows():
                with st.expander(f"🔹 {row['Title']} at {row['Company']} (Cluster {row['Cluster']})"):
                    st.write(f"📍 Location: {row['Location']}")
                    st.write(f"💼 Experience: {row.get('Experience', 'N/A')}")
                    st.write(f"🧠 Skills: {row['Skills']}")
                    st.write(f"📝 Summary: {row.get('Summary', 'No summary available')}")

    # Optionally display all jobs
    st.subheader("📄 All Clustered Jobs")
    st.dataframe(df[["Title", "Company", "Location", "Skills", "Cluster"]], use_container_width=True)
