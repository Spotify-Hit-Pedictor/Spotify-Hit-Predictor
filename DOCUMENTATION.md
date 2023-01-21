## TIMELINE
> 16-10-2023
- Created a Kaggle account for each member
- Downloaded the dataset from Kaggle
- Created a google colab notebook
- Imported dataset (buggy) using wget command
- Imported necessary Libraries
> 17-10-2023
- Imported Dataset (Bug fix)
- Listed out features present in the dataset
- Combined dataset using concat command
- Checked for missing values (none found)
> 18-10-2023
- Learned how to handle missing values
- Learned about class imbalance
- checked for class imbalance (none found)
- Learned about normalization

## Missing Values
**Our dataframe doesn't consists of any missing values**
<br>
The problem of missing value is quite common in many real-life datasets. Missing value can bias the results of the machine learning models and/or reduce the accuracy of the model. 
<br>
Missing data is defined as the values or data that is not stored (or not present) for some variable/s in the given dataset. In Pandas, usually, missing values are represented by NaN. It stands for Not a Number.

### Checking for missing values
The following code finds total number of missing values in the dataframe, 'df'.
<br>
`df.isnull().sum().sum()`

### Handling missing values
- Deleting missing Values (not preferable) (deleting entire row or column)
- Imputing missing values (replacing with arbitary value, mean, mode, median, forward-fill, backward-fill, or Interpolation method)

## Class Imbalance
**Our dataframe doesn't consists of completely balanced data**
<br>
A classification data set with skewed class proportions is called imbalanced. Classes that make up a large proportion of the data set are called majority classes. Those that make up a smaller proportion are minority classes.
20-40% of imbalance is mild.
1-20% of imbalance is moderate.
<1% of imbalance is extreme.

### How it affects accuracy
With class imbalance we can have a super high accuracy just by predicting only majority class.
<br>
So, finding accuracy over an imbalanced data is pointless.

### Ways to fix it
- Use the right evaluation metrics
- Resample the training set
- Use K-fold Cross-Validation in the Right Way
- Ensemble Different Resampled Datasets
- Resample with Different Ratios
- Cluster the abundant class
- Design Your Models

## Data Normalization
Normalization is a technique often applied as part of data preparation for machine learning. The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information.

## Data Visualisation
### Histogram
A histogram is a bar graph-like representation of data that buckets a range of classes into columns along the horizontal x-axis. The vertical y-axis represents the number count or percentage of occurrences in the data for each column. Columns can be used to visualize patterns of data distributions

### Correlation Matrix
A correlation matrix is simply a table which displays the correlation coefficients for different variables. The matrix depicts the correlation between all the possible pairs of values in a table. It is a powerful tool to summarize a large dataset and to identify and visualize patterns in the given data.

## Train Test Split
`train_test_split(X, y, train_size=0.8, random_state=0)`
The train_test_split() method is used to split our data into train and test sets. First, we need to divide our data into features (X) and labels (y). The dataframe gets divided into X_train, X_test, y_train, and y_test. X_train and y_train sets are used for training and fitting the model.

## Model
### Logistic Regression
Logistic regression estimates the probability of an event occurring, such as voted or didn't vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1.

### XGBClassifier
The XGBoost or Extreme Gradient Boosting algorithm is a decision tree based machine learning algorithm which uses a process called boosting to help improve performance. Since it’s introduction, it’s become of one of the most effective machine learning algorithms and regularly produces results that outperform most other algorithms, such as logistic regression, the random forest model and regular decision trees.

### RandomForestClassifier
A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
