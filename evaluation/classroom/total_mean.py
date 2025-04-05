"""Calculate overall mean of distractor probability of all QA model answers

Commandline options:
-o  Evaluate original distractors
-g  Evaluate generated distractors (default)
"""

import pandas as pd
import glob
import argparse


def create_arg_parser():
    """Creates argumentparser and defines command-line options that can be called upon."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--original", action="store_true", help="Use original distractors")
    parser.add_argument("-g", "--generated", action="store_true", help="Use generated distractors")
    return parser.parse_args()


def main():
    # Create argument parser
    args = create_arg_parser()

    if args.original:
        csv_files = glob.glob("original_distractors/*/*summary.csv")
    else:
        csv_files = glob.glob("generated_distractors/*/*summary.csv")
    
    # Load all CSV files and concatenate
    combined_df = pd.concat([pd.read_csv(file, index_col=0) for file in csv_files])

    if args.original:
        distractor_df = combined_df.loc[['original_distractor1', 'original_distractor2', 'original_distractor3']]
    else:
        distractor_df = combined_df.loc[['llama_distractor1', 'llama_distractor2', 'llama_distractor3']]

    # get mean of distractors
    mean_of_distractors = distractor_df.groupby(distractor_df.index).mean().mean(axis=0)
    mean_of_distractors_df = mean_of_distractors.to_frame(name='Overall Mean of Distractor Probability')
    
    # Print and save
    print(mean_of_distractors_df)
    if args.original:
        mean_of_distractors_df.to_csv('original_distractors/_means/_final_mean_distractors.csv')
    else:
        mean_of_distractors_df.to_csv('generated_distractors/_means/_final_mean_distractors.csv')


if __name__ == "__main__":
    main()
