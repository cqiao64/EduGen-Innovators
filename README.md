# EduGen-Innovators

This repository contains code for automated educational test question generation. The project is organized into two main branches:

## BERT Branch
- Contains files for BERT-based evaluation.
- **Automatic_Evaluation.ipynb** implements evaluation metrics such as BERTScore to compare baseline and generated questions.
- **Bert_Score_Testing_GPT2.ipynb** provides additional tools for semantic similarity evaluation.

## Prompt-Chaining Branch
- Contains files for prompt chaining and language model interfacing.
- Files include **finetune_gpt2.py**, **gemma.ipynb**, **Gemma_few_shot.ipynb**, and **Gemma_zero_shot.ipynb**.
- Implements advanced prompting techniques including zero-shot, few-shot, and Chain of Thought strategies, along with training scripts for fine-tuning GPT-2 and Gemma models.
