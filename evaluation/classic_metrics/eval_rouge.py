import pandas as pd
from rouge import Rouge


def compute_rouge_l(reference, candidate, rouge):
    """Function to compute ROUGE-L"""
    scores = rouge.get_scores(candidate, reference)[0]  # ROUGE returns a list
    return scores["rouge-l"]["f"]  # Extract ROUGE-L F1 score


def main():
    # Load CSV
    df = pd.read_csv("data/llama_distractors.csv")  # Change to your file path

    # Display first few rows
    print(df.head())

    # Initialize ROUGE scorer
    rouge = Rouge()

    # Apply ROUGE-L
    df["rouge_l"] = df.apply(lambda row: compute_rouge_l(row["question"], row["option"], rouge), axis=1)

    # Display results
    print(df[["question", "option", "rouge_l", "is_correct"]])

    # Print average scores
    print(df.groupby("is_correct")["rouge_l"].mean())


if __name__ == "__main__":
    main()
