import pandas as pd


def calculate_ttr(text):
    """Calculate Type-Token Ratio (TTR) for a given text."""
    words = text.split()  # Tokenize by splitting on spaces
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0  # Avoid division by zero


def main():
    # Load CSV
    df = pd.read_csv("data/llama_distractors.csv")

    # Combine 'question' and 'option' columns
    df['combined_text'] = df['question'].astype(str) + " " + df['option'].astype(str)

    # Compute TTR for each row
    df['TTR'] = df['combined_text'].apply(calculate_ttr)

    # Overall TTR for the dataset
    overall_ttr = calculate_ttr(" ".join(df['combined_text']))

    # Print results
    print(f"Overall Type-Token Ratio (TTR): {overall_ttr:.4f}")
    print(df[['question', 'option', 'TTR']].head())  # Show sample TTR values per row


if __name__ == "__main__":
    main()
