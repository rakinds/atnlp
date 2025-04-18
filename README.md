# Advanced Topics in Natural Language Processing - Science Quiz Generation
This repository contains all code for Group 4's contribution to the Advanced Topics in Natural Language Processing final project. 

It contains code to generate distractors for questions from the [SciQ dataset](https://huggingface.co/datasets/allenai/sciq) (link to HuggingFace) and evaluate these generated distractors using several evaluation methods: a classic metric-based evaluation using metrics like BLEU, BLEURT and ROUGE, as well as an evaluation method using a 'classroom' of Question-Answering models. 

## Installation
Code can be installed by using `git clone` to clone the repository. 
To install dependencies, create a new Python virtual environment and run:

    pip install -r requirements.txt

## Distractor generation
The distractor generation code is powered by `meta-llama/Llama-2-13b-chat-hf` ([Link](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)). It reads in the SciQ test set and generates distractors for each line of data. Any invalid distractors are discarded and the rest are saved in a csv file to be used for evaluation. 

The distractor code can be run by simply running 

    python3 generator.py

## Evaluation 
### Classic Metrics
The classic metrics that are used to evaluate the generated distractors are BLEU, BLEURT, ROUGE, Jaccard and TTR. Corresponding scripts can be found under `evaluation/classic_metrics/`. These evaluation methods use the data files present in the `evaluation/classic_metrics/data/` folder, which contain reformatted versions of the output from the Llama generation script as well as the original SciQ distractors, for easier processing.

### QA models
The Question Answering models are used to experimentically evaluate the generated distractors. We make use of a set or 'classroom' of MC QA models, which can be run using `classroom.py` under `evaluation/classroom/`. 

<b>Usage example</b>

    python3 classroom.py -o -t hf_aBcDeFgHIJkLmNoPQrsTuVwXYz -m seyonec/ChemBERTa-zinc-base-v1

There are several commandline parameters available:

<b>Commandline options classroom.py:</b>

    -o  Evaluate original distractors
    -g  Evaluate generated distractors (default)
    -m  Define model to use in evaluation
    -t  Add a HuggingFace token (needed for some models)

In this folder, you can also find code to calculate the confidence intervals for each type of QA models' answers as well as code to calculate the overall mean.

<b>Commandline options confidence_analysis.py:</b>

    -o  - Evaluate original distractors
    -g  - Evaluate generated distractors (default)
    -m  - Define model folder to evaluate

<b>Commandline options total_mean.py:</b>

    -o - Evaluate original distractors
    -g - Evaluate generated distractors (default)

