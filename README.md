# DALI 2024 Winter Application - Machine Learning Track
### John Guerrerio

This repository contains my code for the machine learning challenge for the
DALI 2024 winter application.  I chose to analyze the Superstore Dataset.\
\
All files were written in Python 3.10

## Files
- exploratoryAnalysis: Basic exploratory analysis of the superstore dataset to get a better feel for the dataset and get
an idea of any preliminary relationships 
- correlation: Statistical tests and linear regressions to determine if there are and relationships between different features in the dataset
- predictCategory: Code to train and evaluate a deep learning based model to predict product category from the product name.  This would allow us to fill in missing product category (and product sub-category with a few tweaks to the code) cells in the dataset if the product name is available
- profitPrediction: Code to train, fine-tune, and evaluate a Logistic Regression and SVM model to predict if a purchase will make above or below the median profit
- requirements.txt: The libraries and versions I used (for reproducibility)

Please note that each file has detailed documentation; for more information, go to the file of interest

## Things to note:
- After training the model for predictCategory, I ran out of free GPU compute units on Google Colab.  This meant that I couldn't train deep learning models for profitPrediction or complete the optional challenges.
- predictCategory and profitPrediction were developed in google colab where I had access to better computing resources than my personal machine.  As such, the commit history for this repository only shows the addition of these files once I completed them; it does not show the work I put in developing them.
- All package versions are the ones Google Colab uses by default; I have recorded them in requirements.txt for convenience.  I know all the code will run in colab; however, I cannot guarantee it will run in other environments.
- For some reason, GitHub renders my Jupyter Notebook code without newlines.  Please download the files and open them in an IDE - they should render correctly then.