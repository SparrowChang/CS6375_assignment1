#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[2]:


# Step 1: Load the dataset
url = 'https://raw.githubusercontent.com/SparrowChang/CS6375_assignment1/main/auto%2Bmpg/auto-mpg.data'  # Replace with the actual path to the dataset

# Read the CSV file from the URL into a DataFrame
df = pd.read_csv(url, delimiter='\s+', header=None)

# Create a mapping dictionary for column name changes
column_mapping = {0: 'mpg', 
                  1: 'cylinders', 
                  2: 'displacement',
                  3: 'horsepower',
                  4: 'weight',
                  5: 'acceleration',
                  6: 'model year',
                  7: 'origin',
                  8: 'car name'}
# Rename the columns using the mapping dictionary
df = df.rename(columns=column_mapping)


# In[3]:


# Step 2: Pre-processing
# Convert categorical variables to numerical variables (if applicable)
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
# Drop the columns that are not relevant for the regression analysis
del df['car name']
print(df.dtypes)
print(df.shape)

# Create a StandardScaler object
scaler = StandardScaler()
normalized_df = df.copy()
normalized_df.iloc[:, 1:] = scaler.fit_transform(normalized_df.iloc[:, 1:])
# Remove null or NA values
normalized_df = normalized_df.dropna()
# Remove redundant rows
normalized_df = normalized_df.drop_duplicates()
print(normalized_df)


# In[4]:


# Step 2: Pre-processing about abs() correlation matrix
# Calculate the correlation matrix
corr_matrix = normalized_df.corr()

# Take the absolute values of the correlation matrix
abs_corr_matrix = corr_matrix.abs()

# Sort the correlation matrix by a specific column
sort_column = 'mpg'
sorted_abs_corr_matrix = abs_corr_matrix.sort_values(by=sort_column, ascending=False)

# Create a correlation heatmap for the sorted matrix
plt.figure(figsize=(8, 6))
sns.heatmap(sorted_abs_corr_matrix, annot=True, cmap='coolwarm')
plt.title('Sorted Correlation Heatmap')
plt.show


# In[5]:


# Step 3: Split the dataset into training and test sets
train_ratio = 0.8  # Choose the train/test ratio (e.g., 0.8 for 80/20 split)
train_size = int(train_ratio * len(normalized_df))
train_data = normalized_df[:train_size]
test_data = normalized_df[train_size:]


# In[6]:


# Step 4: Implement gradient descent algorithm for linear regression
def gradient_descent(X, y, learning_rate, num_iterations):
    m = len(X)
    n = X.shape[1]
    theta = np.zeros(n)
    cost_history = []

    for iteration in range(num_iterations):
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y
        gradient = np.dot(X.T, loss) / m
        theta -= learning_rate * gradient
        cost = np.sum(loss ** 2) / (2 * m)
        cost_history.append(cost)

    return theta, cost_history


# In[7]:


# Step 5: Evaluate the model on the test set error values for the best set of parameters
def optimize_performance(X_train, y_train, X_test, y_test, learning_rate_range, iteration_range):
    best_error = float('inf')
    best_params = {}

    for learning_rate in learning_rate_range:
        for iterations in iteration_range:
            theta, cost_history = gradient_descent(X_train, y_train, learning_rate, iterations)
            # Calculate predictions using the trained model
            y_pred = np.dot(X_test, theta)

            # Calculate the mean squared error (MSE)
            mse = np.mean((y_pred - y_test) ** 2)

            # Check if this set of parameters gives better performance
            if mse < best_error:
                best_error = mse
                best_params = {'iterations': iterations, 'learning_rate': learning_rate}

    return best_error, best_params


# In[8]:


# Step 5: Evaluate the model on the test set
# Prepare the training data
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
# Add a column of ones for the bias term
X_train = np.column_stack((np.ones(len(X_train)), X_train))

# Prepare the test data
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values
# Add a column of ones for the bias term
X_test = np.column_stack((np.ones(len(X_test)), X_test))

iteration_range = [100, 500, 1000]
learning_rate_range = [0.001, 0.01, 0.1, 0.5]

best_error, best_params = optimize_performance(X_train, y_train, X_test, y_test, learning_rate_range, iteration_range)

print("Best Mean squared error (MSE):", best_error)
print("Best parameters:", best_params)


# In[9]:


# Step 6: Logging and answering the question
log_file_path = 'path_to_log_file.txt'  # Replace with the actual path to the log file

# Write the parameters and error value to the log file
with open(log_file_path, 'w') as log_file:
    log_file.write(f'Best Mean squared error (MSE): {best_error}\n')
    log_file.write(f'Best parameters: {best_params}\n')

# Answering the question
# Whether the best solution is found depends on the obtained MSE and the specific requirements

