import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("llama_distractors.csv")  # Change to your file path

# Display first few rows
print(df.head())

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Function to compute Self-BLEU
def compute_self_bleu(reference, candidate):
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=SmoothingFunction().method1)

# Apply Self-BLEU
df["self_bleu"] = df.apply(lambda row: compute_self_bleu(row["question"], row["option"]), axis=1)

# Function to save results to an Excel file
def save_to_excel(df, filename="self_bleu_results_llama.xlsx"):
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")

# Save results to Excel
save_to_excel(df)

# Display results
print(df[["question", "option", "self_bleu", "is_correct"]])

# Plot Self-BLEU score distribution
sns.boxplot(x=df["is_correct"], y=df["self_bleu"])
plt.title("Self-BLEU Score Distribution for Correct vs. Incorrect Answers")
plt.show()

# Print average scores
print(df.groupby("is_correct")["self_bleu"].mean())

