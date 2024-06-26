import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and Inspect the Dataset
# Download historical data for a specific stock (e.g., Apple Inc.)
stock_symbol = 'TSLA'
data = yf.download(stock_symbol, start='2020-01-01', end='2023-01-01')

# Create technical indicators
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
data['RSI'] = 100 - (100 / (1 + data['Close'].diff().rolling(14).apply(lambda x: (x[x > 0].sum() / (-x[x < 0].sum())), raw=True)))
data['Volume'] = data['Volume']

# Generate the target variable: 1 if the price went up the next day, 0 if it went down
data['Price Movement'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop rows with missing values
data = data.dropna()

# Inspect the dataset
print(data.head())

# Step 2: Data Preprocessing
# Feature selection
features = ['Close', 'MA50', 'MA200', 'RSI', 'Volume']
X = data[features]
y = data['Price Movement']

# Normalizing numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Training the Model
model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Using RBF kernel
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Visualizing the Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 6: Interpreting the Results
# Understanding the impact of features can be done by analyzing the support vectors and decision function.
# For simplicity, we print out some key results.
support_vectors = model.support_vectors_
print(f'Number of support vectors: {len(support_vectors)}')
