"""
Calculate the confidence intervals for QA models' answers

Commandline options:
-o  Evaluate original distractors
-g  Evaluate generated distractors (default)
-m  Define model folder to evaluate

List of model folders:
albert-base-v2
albert-large-v2
bert-base-uncased
bert-large-uncased
biobert-base-cased-v1.2
BioLinkBERT-base
biomed_roberta_base
BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
ChemBERTa-zinc-base-v1
deberta-v3-base
deberta-v3-large
distilbert-base-uncased
roberta-base
roberta-large
scibert_scivocab_uncased
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import glob
import argparse


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval"""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2, n-1) if n > 1 else 0
    return mean, mean - h, mean + h


def create_arg_parser():
    """Creates argumentparser and defines command-line options that can be called upon."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--original", action="store_true", help="Use original distractors")
    parser.add_argument("-g", "--generated", action="store_true", help="Use generated distractors")
    parser.add_argument("-m", "--model", type=str, help="Fill in the model identifier to evaluate")
    return parser.parse_args()


def main():
    # Create argument parser
    args = create_arg_parser()

    if not args.model:
        print('Exit: Please define a model using the flag -m [model name]')
        exit()

    # File path
    if args.original:
        folder = "original_distractors"
    else:
        folder = "generated_distractors"

    model_name = args.model
    path_to_csv_files = folder + "/" + model_name + "/*.csv"
    
    # Load and combine CSV files
    all_files = glob.glob(path_to_csv_files)
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
    correct_probs = df[df["label"] == "correct"]["probability"]

    if args.original:
        distractor_labels = ['original_distractor1', 'original_distractor2', 'original_distractor3']
    else:
        distractor_labels = ["llama_distractor1", "llama_distractor2", "llama_distractor3"]

    distractor_probs = {
        distractor: df[df["label"] == distractor]["probability"]
        for distractor in distractor_labels
    }
    
    # Calculate confidence intervals
    ci_results = {}
    ci_results['correct'] = calculate_confidence_interval(correct_probs)
    
    for distractor, probs in distractor_probs.items():
        ci_results[distractor] = calculate_confidence_interval(probs)
    
    # Format and print results
    ci_df = pd.DataFrame(ci_results, index=['Mean Probability', 'CI Lower Bound', 'CI Upper Bound']).T
    print(ci_df)
    
    # Save to CSV
    output_filename = f"{folder}/{model_name}/{model_name}_confidence_intervals_summary.csv"
    ci_df.to_csv(output_filename)


if __name__ == "__main__":
    main()
