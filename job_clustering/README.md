# Job Clustering & Alert System

This project scrapes job listings from [karkidi.com](https://www.karkidi.com), clusters them based on required skills using unsupervised learning, and alerts users when new jobs match their preferred skill clusters. It also includes a Streamlit web app for interactive browsing.

---

## Folder Structure

job_clustering/

├── models/

│ ├── tfidf_vectorizer.pkl   # Saved TF-IDF vectorizer model

│ └── kmeans_model.pkl   # Saved KMeans clustering model

├── karkidi_jobs.csv   # Raw scraped jobs data CSV

├── karkidi_clustered.csv   # Baseline clustered jobs CSV

├── karkidi_clustered_new.csv   # Latest clustered jobs CSV (new data)

├── scrape_jobs.py   # Script to scrape job listings

├── clustering_classify_useralert.py   # Script to perform clustering, save models, and alert users

└── app.py   # Streamlit web app to explore jobs

## How to Use

Scrape jobs and save raw data

python scrape_jobs.py

This saves scraped job listings to karkidi_jobs.csv.

Cluster jobs, save models, and alert users

python clustering_classify_useralert.py

Trains TF-IDF vectorizer and KMeans clustering model (if not already saved).

Assigns clusters to jobs and saves updated CSV files.

Alerts if new jobs match user-preferred clusters (configured in clustering_classify_useralert.py).

Launch Streamlit app to explore jobs

streamlit run app.py

## Configuration

Set user preferred clusters in clustering_classify_useralert.py (variable USER_PREFERRED_CLUSTERS) for alerts.

Change search keyword in scrape_jobs.py if you want to scrape different types of jobs.

