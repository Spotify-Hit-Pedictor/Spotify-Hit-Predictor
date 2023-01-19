# TIMELINE
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
