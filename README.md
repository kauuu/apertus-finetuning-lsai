# Evaluating Apertus Models for Multilingual Legal Summarisation of Swiss Supreme Court Decisions

This repository consists of the code for finetuning Apertus models and evaluating their performance on ![Swiss Landmark Decisions Summarisation](https://huggingface.co/datasets/ipst/slds) (SLDS). Each of the modules was adapted from the following existing codebases: 

- Finetuning Recipe: https://github.com/swiss-ai/apertus-finetuning-recipes
- SLDS Evaluation - https://github.com/rolshoven/slds-eval

The contributions include: 
1. Perform LoRA and Full Finetuning (FFT) on the SLDS dataset of Apertus Family of Models (8B and 70B).
2. Compare their performance with other open-source models using automatic metrics and LLM-as-a-Judge.
3. Reason the results and provide future suggestions.

The report of the work can be found in the repository. 

## Task Description

Swiss Landmark Decision Summarisation is a cross-lingual task, where given a lengthy Landmark Decision document in German/Italian/French, the model is required to generate a concise headnote in one of the three languages (DE, FR, IT), containing the key legal points discussed in the decision, along with how they were interpreted. 
This task is challenging due to its complicated understanding of legal jargon, along with the need for a structured and concise output. 

## Our codebase explained

- `/finetuning` contains the code needed to submit jobs to finetune models. The config file (`/finetuning/configs`)can be modified to control the hyperparameters, dataset and the model to finetune.
- `/evaluation` contains the code to run the evaluation of our model on the SLDS dataset. It returns the BERTScore, ROUGE, BLEU and LLM-as-a-Judge scores.

Each of the above two repositories contains the original README, indicating how to run them. 
