"""
Calculate the confidence intervals for QA models' answers
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import glob


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval"""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2, n-1) if n > 1 else 0
    return mean, mean - h, mean + h


def main():
    # File path
    model_name = 'albert_base' # change to evaluate different QA model answers
    path_to_csv_files = model_name + '/*.csv'
    
    # Load and combine CSV files
    all_files = glob.glob(path_to_csv_files)
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
    correct_probs = df[df['label'] == 'correct']['probability']

    # Uncomment right line to evaluate either llama (generated) distractors or original distractors
    distractor_labels = ['llama_distractor1', 'llama_distractor2', 'llama_distractor3']
    #distractor_labels = ['original_distractor1', 'original_distractor2', 'original_distractor3']

    distractor_probs = {
        distractor: df[df['label'] == distractor]['probability']
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
    ci_df.to_csv('results/confidence_intervals_summary.csv')


if __name__ == "__main__":
    main()
