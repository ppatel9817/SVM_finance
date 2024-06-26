# SVM_finance

This project demonstrates how to use Support Vector Machines (SVM) to predict short-term stock price movements. We use historical stock data for Tesla (TSLA) to build a model that predicts whether the stock price will go up or down the next day.

# Installation

To run this project, you need to have Python installed along with the following libraries:

pandas
numpy
yfinance
scikit-learn
matplotlib
seaborn

# Usage

Load and Inspect the Dataset:
Download historical data for Tesla (TSLA) and create technical indicators like moving averages (MA50, MA200) and Relative Strength Index (RSI).

Data Preprocessing:
Select relevant features and normalize them.

Splitting the Data:
Split the dataset into training and testing sets.

Training the Model:
Train an SVM model using the training data.

Model Evaluation:
Evaluate the model’s performance using accuracy, precision, recall, and a confusion matrix.

Interpreting the Results:
Analyze the model’s predictions and support vectors.

# Results

The model’s performance metrics are printed after the evaluation step:

Accuracy: Proportion of correct predictions
Precision: Proportion of true positive predictions out of all positive predictions
Recall: Proportion of true positive predictions out of all actual positive cases
Confusion Matrix: Breakdown of true positives, true negatives, false positives, and false negatives

# Interpretation

The confusion matrix and performance metrics provide insights into the model’s effectiveness. A higher number of support vectors might indicate complexity in the data.
