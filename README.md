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
The classic metrics that are used to evaluate the generated distractors are BLEU, BLEURT, ROUGE, Jaccard and TTR. Corresponding scripts can be found under `evaluation/classic_metrics/`. 

### QA models
The Question Answering models are used to experimentically evaluate the generated distractors. We make use of a set or 'classroom' of MC QA models, which can be run using `classroom.py` under `evaluation/classroom/`. Simply change the `model_name` variable to evaluate a certain QA model and run the script. 

In this folder, you can also find code to calculate the confidence intervals for the QA models' answers as well as code to calculate the overall mean.
