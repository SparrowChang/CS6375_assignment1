#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score


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


# Step 4: Train the linear regression model
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

model = LinearRegression()
model.fit(X_train, y_train)


# In[7]:


# Step 5: Evaluate the model on the test set
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)


# In[8]:


# Step 6: Logging and output
log_file_path = 'path_to_log_file_MLlib.txt'  # Replace with the actual path to the log file

# Write the parameters and error values to the log file
with open(log_file_path, 'w') as log_file:
    log_file.write(f'Mean squared error (MSE): {mse}\n')
    log_file.write(f'R2 score: {r2}\n')
    log_file.write(f'Explained variance: {explained_var}\n')

# Output weight coefficients
weights = model.coef_
intercept = model.intercept_
print('Weight Coefficients:', weights)
print('Intercept:', intercept)

# Output additional evaluation statistics
print('MSE:', mse)
print('R2 Score:', r2)
print('Explained Variance:', explained_var)

# Repeat the above steps with different parameter values to check for better solutions
# Compare the performance metrics and choose the best set of parameters

# Answering the question
# Whether the package has found the best solution depends on the performance metrics and the specific requirements

