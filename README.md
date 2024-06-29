# DALI Machine Learning Challenge

This repository contains my code for the DALI machine learning challenge.  I chose to analyze the Superstore Dataset.

## Files
- exploratoryAnalysis: Exploratory analysis of the superstore dataset to get a better feel for the dataset.  I also use heatmaps, linear regression, and statistical tests to gain insight into relationships between different features.
- predictCategory: Code to train and evaluate a BERT-based model to predict product category from the product name. This would allow us to fill in missing product category (and product sub-category with a few tweaks to the code) cells in the dataset if the product name is available.  I also used shapley values to increase the interpretability of the BERT model.
- profitPrediction: Code to train, fine-tune, and evaluate a Logistic Regression, SVM, and XGBoost model to predict if a purchase will make above or below the median profit.  I applied shapley values to increase the interpretability of each model and deployed the best-performing model of each type to a huggingface space.
- requirements.txt: The libraries and versions I used (for reproducibility)

Please note that each file has detailed documentation; for more information, go to the file of interest

## Things to note:
- All files were written in Python 3.10
- All package versions are the ones Google Colab uses by default; I have recorded them in requirements.txt for convenience.  Google colab is a good environment to run this code in, especially for predictCategory which trains a BERT-based classifier and benefits greatly from a GPU.
