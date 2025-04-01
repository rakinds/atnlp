""" 
A program that generates distractors using data from the SciQ dataset.

Takes the question, context, and correct answer of every line of data from the SciQ dataset test set.
Creates a prompt with this information and generates 3 new distractors using the created prompt and Llama-2-13b-chat-hf.
The Llama output is run through a cleaner and the 3 isolated distractors are added to a csv file together with the original question and correct answer.  
"""

from transformers import AutoModelForMultipleChoice, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from random import shuffle
import torch
from datasets import load_dataset
import outlines
import re


def distractor_prompt(support, question, correct_answer):
    """Create prompt using question, support and correct answer"""
    return f"""[INST] You are provided with context, a question, and the correct answer.
    
        Question: {question}
        
        Context: {support}
        
        Correct Answer: {correct_answer}
        
        Output exactly three incorrect answers separated by commas.
        Do NOT write any introduction, labels, or extra text. ONLY write the three incorrect answers in this format:
        answer1, answer2, answer3
        [/INST]"""


def main():
    # Define model and tokenizer
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    llm = outlines.models.transformers(
        model_name,
        device="auto",
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    
    # Change value here for the amount of distractor lines made
    dataset = load_dataset("allenai/sciq", split="test")
    
    # Define Outlines generation parameters
    generator = outlines.generate.regex(
        llm,
        r"[^,\n]+,[^,\n]+,[^,\n]+"
    )
    
    # Output file
    output_csv = "generated_distractors.csv"
    
    # Store rows to save later
    rows = []
    interval = 1
    
    for example in dataset:
        print(interval)
        prompt = distractor_prompt(
            example["support"],
            example["question"],
            example["correct_answer"]
        )
    
        # Get model output
        raw_output = generator(prompt)
    
        # Clean model output
        cleaned = raw_output.strip().lower()
        cleaned = re.sub(r"^sure[^:]*:?", "", cleaned)
        cleaned = re.sub(r"here (they|they are|are|they are as follows):?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[;|]", ",", cleaned)
        cleaned = re.sub(r"[^\w\s,-]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+\band\b\s+", ", ", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip(" .,")
    
        distractors = [re.sub(r"^\d+\.\s*", "", d).strip() for d in cleaned.split(",") if d.strip()] 

        # Skip line if distractors cannot be cleaned
        if len(distractors) != 3:
            print(f"Skipping example due to bad output: {raw_output}")
            continue
    
        row = {
            "question": example["question"],
            "correct_answer": example["correct_answer"],
            "llama_distractor1": distractors[0],
            "llama_distractor2": distractors[1],
            "llama_distractor3": distractors[2],
            "original_distractor1": example["distractor1"],
            "original_distractor2": example["distractor2"],
            "original_distractor3": example["distractor3"],
        }
        rows.append(row)
        interval += 1
    
    # Save results to CSV
    saved_name = "results/generated_distractors.csv"
    docs = pd.DataFrame(rows)
    docs.to_csv(saved_name, index=False)
    print(f"Cleaned distractors saved to {saved_name}")
    docs.head()


if __name__ == "__main__":
    main()
