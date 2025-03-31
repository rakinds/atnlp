# Advanced Topics in Natural Language Processing - Science Quiz Generation
This repository contains all code for Group 4's contribution to the Advanced Topics in Natural Language Processing final project. 

It contains code to generate distractors for questions from the [SciQ dataset](https://huggingface.co/datasets/allenai/sciq) (link to HuggingFace) and evaluate these generated distractors using several evaluation methods: a classic metric-based evaluation using metrics like BLEU, BLEURT and ROUGE, as well as an evaluation method using a 'classroom' of Question-Answering models. 

## Installation
Code can be installed by using `git clone` to clone the repository. 
To install dependencies, create a new Python virtual environment and run:

    pip install -r requirements.txt

## Distractor generation
The distractor generation code is powered by `meta-llama/Llama-2-13b-chat-hf` ([Link](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)). 

## Evaluation 
### Classic Metrics


### QA models
