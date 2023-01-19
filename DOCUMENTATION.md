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

## Missing Values
The problem of missing value is quite common in many real-life datasets. Missing value can bias the results of the machine learning models and/or reduce the accuracy of the model. <br>
Missing data is defined as the values or data that is not stored (or not present) for some variable/s in the given dataset. In Pandas, usually, missing values are represented by NaN. It stands for Not a Number.

### Checking for missing values
The following code finds total number of missing values in the dataframe, 'df'.<br>
`df.isnull().sum().sum()`

