"""Evaluate generated distractors using BLEURT"""

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score


def compute_bleurt(reference, candidate, tokenizer, model):
    """Function to compute BLEURT score"""
    inputs = tokenizer(reference, candidate, return_tensors="pt")
    with torch.no_grad():
        score = model(**inputs).logits.squeeze().item()
    return score


def main():
    # Load CSV
    df = pd.read_csv("data/llama_distractors.csv")
    
    # Display first few rows
    print(df.head())

    # Load BLEURT
    token = "" # add HF token here to use this model
    tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512", token=token)
    model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512", token=token)

    # Apply BLEURT to all rows
    df["bleurt_score"] = df.apply(lambda row: compute_bleurt(row["question"], row["option"], tokenizer, model), axis=1)
    
    # Display results
    print(df[["question", "option", "bleurt_score", "is_correct"]])
    
    # Print average scores
    print(df.groupby("is_correct")["bleurt_score"].mean())
    
    # Convert BLEURT scores into predictions (Threshold tuning may be needed)
    df["bleurt_predicted"] = df["bleurt_score"] > df["bleurt_score"].median()
    
    # Calculate metrics
    accuracy = accuracy_score(df["is_correct"], df["bleurt_predicted"])
    precision = precision_score(df["is_correct"], df["bleurt_predicted"])
    recall = recall_score(df["is_correct"], df["bleurt_predicted"])
    
    print(f"Accuracy: {accuracy: .2f}, Precision: {precision: .2f}, Recall: {recall: .2f}")


if __name__ == "__main__":
    main()
