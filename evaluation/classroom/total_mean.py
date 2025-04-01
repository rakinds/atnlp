"""Calculate overall mean of distractor drobability of all QA model answers"""

import pandas as pd
import glob


def main():
    csv_folder = '_means/*.csv'
    
    # Load all CSV files and concatenate 
    csv_files = glob.glob(csv_folder)
    combined_df = pd.concat([pd.read_csv(file, index_col=0) for file in csv_files])

    # Comment right line to evaluate llama or original distractors
    distractor_df = combined_df.loc[['llama_distractor1', 'llama_distractor2', 'llama_distractor3']]
    #distractor_df = combined_df.loc[['original_distractor1', 'original_distractor2', 'original_distractor3']]

    # get mean of distractors
    mean_of_distractors = distractor_df.groupby(distractor_df.index).mean().mean(axis=0)
    mean_of_distractors_df = mean_of_distractors.to_frame(name='Overall Mean of Distractor Probability')
    
    # Print and save
    print(mean_of_distractors_df)
    mean_of_distractors_df.to_csv('final_mean_distractors.csv')


if __name__ == "__main__":
    main()
