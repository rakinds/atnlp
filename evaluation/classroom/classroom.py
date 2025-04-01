"""
Use Question answering model to evaluate distractors.

The newly generated or original distractors + question + correct answer are given to the BERT model which does the following:
1. Calculates the model"s confidence (softmax probability) for each option
2. Checks if the model chose the correct answer
3. Records the confidence and prediction for each option
4. Counts how often a distractor was ranked above the correct answer
5. Saves the results to a CSV file for further analysis

Model options:
* albert/albert-base-v2
* albert/albert-large-v2
* google-bert/bert-base-uncased
* google-bert/bert-large-uncased
* dmis-lab/biobert-base-cased-v1.2
* michiyasunaga/BioLinkBERT-base
* allenai/biomed_roberta_base
* microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
* seyonec/ChemBERTa-zinc-base-v1
* microsoft/deberta-v3-base
* microsoft/deberta-v3-large
* distilbert/distilbert-base-uncased
* FacebookAI/roberta-base
* FacebookAI/roberta-large
* allenai/scibert_scivocab_uncased
"""

from transformers import AutoModelForMultipleChoice, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from random import shuffle
import torch
from datasets import load_dataset
import outlines
import re


def main():
    # Read in and look at text column of data csv
    docs = pd.read_csv("835_test_distractors.csv")
    docs.head()

  # Load BERT model
    model_name = "seyonec/ChemBERTa-zinc-base-v1" # Change model name here 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)

    correct = 0
    a_true = []
    a_predicted = []

    eval_rows = []  # store rows for csv analysis

    # Comment / uncomment the right line to evaluate llama distractors or original distractors
    confusion_counts = {"llama_distractor1": 0, "llama_distractor2": 0, "llama_distractor3": 0}
    #confusion_counts = {"original_distractor1": 0, "original_distractor2": 0, "original_distractor3": 0}

    for i in range(len(docs)):
        # Extract question and correct answer
        q = docs.loc[i, "question"]
        correct_answer = docs.loc[i, "correct_answer"]

        # Extract options and shuffle
        options = [
            docs.loc[i, "llama_distractor1"], # change to original_distractor1 to evaluate original distractors
            docs.loc[i, "llama_distractor2"], # change to original_distractor2 to evaluate original distractors
            docs.loc[i, "llama_distractor3"], # change to original_distractor3 to evaluate original distractors
            correct_answer
        ]

        # Comment / uncomment the right line to evaluate llama distractors or original distractors
        option_labels = ["llama_distractor1", "llama_distractor2", "llama_distractor3", "correct"]
        #option_labels = ['original_distractor1', 'original_distractor2', 'original_distractor3', 'correct']

        paired_options = list(zip(options, option_labels))
        shuffle(paired_options)

        shuffled_options, shuffled_labels = zip(*paired_options)

        # Format input as (question, option) pairs
        input_pairs = [[q, opt] for opt in shuffled_options]
        inputs = tokenizer(input_pairs, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}  

        # Run prediction
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0].tolist()
        predicted_index = int(torch.argmax(logits, dim=1).item())
        predicted_answer = shuffled_options[predicted_index]

        # Evaluate whether correct
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1

        a_true.append(correct_answer)
        a_predicted.append([predicted_answer, probs[predicted_index]])

        # Save full row for external analysis
        for j, (opt, label) in enumerate(zip(shuffled_options, shuffled_labels)):
            eval_rows.append({
                "question": q,
                "option": opt,
                "label": label,
                "is_correct": (opt == correct_answer),
                "probability": probs[j],
                "predicted": (j == predicted_index)
            })

        # Track if a distractor was ranked above the correct answer
        ranked = sorted(zip(shuffled_labels, probs), key=lambda x: x[1], reverse=True)
        for rank, (label, _) in enumerate(ranked):
            if label == "correct":
                break
            elif label.startswith("d"):
                confusion_counts[label] += 1
                break

    # Report
    accuracy = correct/len(docs)
    accuracy_percent = f"{accuracy:.2%}"

    print(f"Model: {model_name}")
    print(f"Accuracy: {correct}/{len(docs)} ({accuracy_percent})")
    print("Sample predictions and confidence:\n", a_predicted[:5])

    # get the right model name
    model_identifier = model_name.split("/")[-1]
    accuracy_filename = accuracy_percent.replace("%", "pct")

    output_filename = f"{model_identifier}_distractor_eval_acc={accuracy_filename}.csv"
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(output_filename, index=False)
    print(f"Saved predictions to {output_filename}.")


if __name__ == "__main__":
    main()
