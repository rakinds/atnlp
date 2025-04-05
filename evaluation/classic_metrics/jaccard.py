import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import edit_distance

# Load CSV
df = pd.read_csv("llama_distractors.csv")  # Change to your file path

# Display first few rows
print(df.head())

# Function to compute Jaccard Similarity
def jaccard_similarity(reference, candidate):
    reference_tokens = set(reference.lower().split())
    candidate_tokens = set(candidate.lower().split())
    return len(reference_tokens & candidate_tokens) / len(reference_tokens | candidate_tokens) if reference_tokens | candidate_tokens else 0

# Function to compute Cosine Similarity
def cosine_sim(reference, candidate):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference, candidate])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

# Apply similarity metrics
df["jaccard_similarity"] = df.apply(lambda row: jaccard_similarity(row["question"], row["option"]), axis=1)
df["edit_distance"] = df.apply(lambda row: edit_distance(row["question"].lower(), row["option"].lower()), axis=1)
df["cosine_similarity"] = df.apply(lambda row: cosine_sim(row["question"], row["option"]), axis=1)

# Display results
print(df[["question", "option", "jaccard_similarity", "edit_distance", "cosine_similarity", "is_correct"]])

# Print average scores
print(df.groupby("is_correct")[["jaccard_similarity", "edit_distance", "cosine_similarity"]].mean())
