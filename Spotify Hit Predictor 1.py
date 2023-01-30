import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

decades = ["60", "70", "80", "90", "00", "10"]
df = [pd.read_csv("C:/Users/HP/Documents/Spotify Hit Predictor Dataset/dataset-of-" + i + "s.csv") for i in decades]
for i, decade in enumerate([1960, 1970, 1980, 1990, 2000, 2010]):
    df[i]['decade'] = pd.Series(decade, index=df[i].index)
df = pd.concat(df).reset_index(drop=True)
# print(df.head())

# print(df.isnull().sum())

null_col = [col for col in df.columns if None in df[col]]
# print(null_col)

num_col = [col for col in df.columns if df[col].dtypes != "object"]
# print(num_col)

dis_col = [col for col in num_col if len(df[col].unique()) < 15]
# print(dis_col)

for col in dis_col:
    df.groupby(col)["target"].median().plot.bar()
    plt.xlabel(col)
    plt.ylabel("target")
    # plt.show()

cont_col = [col for col in num_col if col not in dis_col]
# print(cont_col)

for col in cont_col:
    df[col].hist(bins=20)
    plt.xlabel(col)
    plt.ylabel("target")
    # plt.show()

cat_col = [col for col in df.columns if df[col].dtypes == "object"]
# print(cat_col)

df.drop(labels=cat_col, axis=1, inplace=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
corr_matrix = df.corr()
# print(corr_matrix)
sns.heatmap(df.corr())
# plt.show()

X = df.drop(labels=["target"], axis=1)
y = df["target"]

X.drop(["chorus_hit", "sections"], axis=1, inplace=True)
scaler = MinMaxScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)
# print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

model = RandomForestClassifier(n_estimators=1000, max_depth=24, random_state=0, n_jobs=8)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(score)
