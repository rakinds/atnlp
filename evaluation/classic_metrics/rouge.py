import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rouge import Rouge

# Load CSV
df = pd.read_csv("llama_distractors.csv")  # Change to your file path

# Display first few rows
print(df.head())

# Initialize ROUGE scorer
rouge = Rouge()

# Function to compute ROUGE-L
def compute_rouge_l(reference, candidate):
    scores = rouge.get_scores(candidate, reference)[0]  # ROUGE returns a list
    return scores["rouge-l"]["f"]  # Extract ROUGE-L F1 score

# Apply ROUGE-L
df["rouge_l"] = df.apply(lambda row: compute_rouge_l(row["question"], row["option"]), axis=1)

# Display results
print(df[["question", "option", "rouge_l", "is_correct"]])

# Print average scores
print(df.groupby("is_correct")["rouge_l"].mean())
