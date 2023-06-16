# CS6375_assignment1
CS6375 assignment1 files: 
1. 'part1.py': Linear Regression using Gradient Descent Coding in Python.
Run all, it will generate 'path_to_log_file.txt'
 
2. 'part2.py': Linear Regression using ML libraries.
Run all, it will generate 'path_to_log_file_MLlib.txt'

3. 'CS6375 Assignment1_230616_final.pdf': details as below.

---------------------------
# 'CS6375 Assignment1_230616_final.pdf'

# 1 Linear Regression using Gradient Descent Coding in Python (75 points)
1. Choose a dataset suitable for regression from UCI ML Repository: -https://archive.ics.uci.edu/ml/datasets.php. 
A: url='https://raw.githubusercontent.com/SparrowChang/CS6375_assignment1/main/auto%2Bmpg/auto-mpg.data', we use the dataset (auto MPG from UCI ML repository) and upload on Ching-Yi’s Github. 

2. Pre-process your dataset. Pre-processing includes the following activities:
A: 
-Rename column name by mapping dictionary (ex: mpg, cylinders, displacement...)
-Change data structures from object to numerical values (ex: horsepower)
-Drop the columns that are not relevant for the regression analysis (ex: car name)
-Normalize data (ex: StandardScaler())
-Remove null or NA values (ex: dropna())
-Remove redundant rows (ex: drop_duplicates())
-Draw a Correlation Heatmap. Y is the output, ‘mpg’ (mile per gallon), we could know the attitudes ‘weight’, ‘displacement’, ‘horsepower’ and ‘cylinders’ show higher correlation coefficient with the output ‘mpg’.

3. After pre-processing split the dataset into training and test parts. It is up to you to choose the train/test ratio
A: We choose train/test ratio = 80/20

4. Use the training dataset to construct a linear regression model. 
A: define gradient descent function with the learning rate and iterations

5. Apply the model you created in the previous step to the test part of the dataset. Report the test dataset error values for the best set of parameters obtained from the previous part. 
A: Consider 7 multiple attributes and think of the vector from the weight update equation. 
We tune these parameters to achieve the optimum error value. 
iteration_range = [100, 500, 1000]
learning_rate_range = [0.001, 0.01, 0.1, 0.5]
We create a log file that indicates the parameters used to get the best error (MSE). 
In log file (optimum test error value)
Best Mean squared error (MSE): 27.518418930391345
Best parameters: {'iterations': 1000, 'learning_rate': 0.1}

6. Answer this question: Are you satisfied that you have found the best solution? Explain.
A: Yes, we satisfied the result, MSE of gradient descent is around 27.51 which is close to MSE of ML libraries (sklearn.linear_model LinearRegression) 27.46.

# 2 Linear Regression using ML libraries (25 points)
A: From step 1 to step 3 is exactly the same as the 1st part.  

4. The big difference from the 1st part, we use any ML library that performs linear regression from Scikit Learn package. https://scikit-learn.org
A: Use sklearn ML library - LinearRegression().

5. Apply the model you created in the previous step to the test part of the dataset. Report the test dataset error values for the best set of parameters obtained from the previous part. 
A: We create a log file that indicates the parameters used and the error (MSE) value.
In log file (optimum test error value)
Mean squared error (MSE): 27.46273183235903
R2 score: 0.2414371494496731
Explained variance: 0.4305663485275186

6. Answer this question: Are you satisfied that you have found the best solution? Explain.
A: Yes, MSE is around 27.46 which is close to the gradient descent method. 
