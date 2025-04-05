import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import edit_distance


def jaccard_similarity(reference, candidate):
    """Function to compute Jaccard Similarity"""
    reference_tokens = set(reference.lower().split())
    candidate_tokens = set(candidate.lower().split())
    return len(reference_tokens & candidate_tokens) / len(reference_tokens | candidate_tokens) if reference_tokens | candidate_tokens else 0


def cosine_sim(reference, candidate):
    """Function to compute Cosine Similarity """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference, candidate])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]


def main():
    # Load CSV
    df = pd.read_csv("data/llama_distractors.csv")

    # Display first few rows
    print(df.head())

    # Apply similarity metrics
    df["jaccard_similarity"] = df.apply(lambda row: jaccard_similarity(row["question"], row["option"]), axis=1)
    df["edit_distance"] = df.apply(lambda row: edit_distance(row["question"].lower(), row["option"].lower()), axis=1)
    df["cosine_similarity"] = df.apply(lambda row: cosine_sim(row["question"], row["option"]), axis=1)

    # Display results
    print(df[["question", "option", "jaccard_similarity", "edit_distance", "cosine_similarity", "is_correct"]])

    # Print average scores
    print(df.groupby("is_correct")[["jaccard_similarity", "edit_distance", "cosine_similarity"]].mean())


if __name__ == "__main__":
    main()
