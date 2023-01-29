import pandas as pd
import numpy as np
import matplotlib as pt
import seaborn as sb

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score

from matplotlib import pylab as plb
from matplotlib import pyplot as plt

pt.style.use('ggplot')
sb.set_style('white')
plb.rcParams['figure.figsize'] = 12, 8

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

decades = ["60", "70", "80", "90", "00", "10"]
df = [pd.read_csv("C:/Users/HP/Documents/Spotify Hit Predictor Dataset/dataset-of-" + i + "s.csv") for i in decades]
for i, decade in enumerate([1960, 1970, 1980, 1990, 2000, 2010]):
    df[i]['decade'] = pd.Series(decade, index=df[i].index)
df = pd.concat(df).reset_index(drop=True)

# print(df.head(10))
# print(df.isnull().sum())
# print(df.describe(include='all'))

# Feature Engineering

# track
# print(len(df["track"].unique()))
df.drop(labels=["track"], axis=1, inplace=True)

# uri
# print(len(df["uri"].unique()))
df.drop(labels=["uri"], axis=1, inplace=True)

# artist
# print(len(df["artist"].unique()))
artist = df["artist"].value_counts() > 10
# print(len(artist))
df["artist"] = df["artist"].apply(lambda x: "Unknown" if not artist.loc[x] else x)
# print(df["artist"].value_counts())
# print(df.describe(include="all"))
# print(len(df["artist"].unique()))

artist_col = df["artist"]
df = pd.get_dummies(data=df, columns=["artist"])
df.drop(labels=["artist_Unknown"], axis=1, inplace=True)
# print(df.columns)

artist_target = pd.crosstab(index=artist_col, columns=df["target"])
# print(artist_target)
# print(artist_target.loc[(artist_target[0] != 0) & (artist_target[1] != 0)])

flop_artist_col = "artist_" + artist_target.loc[artist_target[0] == 0].index
flop_artist = df[flop_artist_col].sum(axis=1)
# print(sum(flop_artist == 1))

hit_artist_col = "artist_" + artist_target.loc[artist_target[1] == 0].index
hit_artist = df[hit_artist_col].sum(axis=1)
# print(sum(hit_artist == 1))

df = pd.concat([df, hit_artist, flop_artist], axis=1)
# print(df.head())

df.drop(labels=flop_artist_col, axis=1, inplace=True)
df.drop(labels=hit_artist_col, axis=1, inplace=True)
df.rename(columns={0: "flop_artist", 1: "hit_artist"}, inplace=True)

# print(df.columns)
# print(df.describe(include="all"))
# print(df.head())

# danceability
# df["danceability"].plot()
# plt.show()

# print(sum(df["danceability"] <= 0.10))
# print(sum(df["danceability"] >= 0.90))

df.drop(labels=df[df["danceability"] <= 0.10].index, axis=0, inplace=True)
df.drop(labels=df[df["danceability"] >= 0.90].index, axis=0, inplace=True)
min_scaler = MinMaxScaler()
# print(df.describe())
min_scaled = min_scaler.fit_transform(df["danceability"].to_numpy().reshape(-1, 1))
# print(min_scaled)
# print(len(df["danceability"]))
df["danceability"] = min_scaled
# print(df.shape)
# print(df["danceability"].describe())
# df["danceability"].plot()
# plt.show()

# energy
# df["energy"].plot()
# plt.show()

# print(df["energy"].describe())
# print(sum(df["energy"] <= 0.10))
# print(sum(df["energy"] >= 0.96))

df.drop(labels=df[df["energy"] <= 0.10].index, axis=0, inplace=True)

# print(df["energy"].describe())
min_scaled = min_scaler.fit_transform(df["energy"].to_numpy().reshape(-1, 1))
df["energy"] = min_scaled
# print(df["energy"].describe())

# df["energy"].plot()
# plt.show()

# key
# df["key"].plot()
# plt.show()

# print(df["key"].unique())
key_target = pd.crosstab(index=df["key"], columns=df["target"])
# print(key_target)
#
df = pd.get_dummies(data=df, columns=["key"])
df.drop(labels=["key_11"], axis=1, inplace=True)
# print(df.columns)
# print(df.shape)

# loudness
# df["loudness"].plot()
# plt.show()

# print(df["loudness"].describe())
# print(sum(df["loudness"] >= 0))
# print(sum(df["loudness"] <= -40))

df.drop(labels=df[df["loudness"] >= 0].index, axis=0, inplace=True)
df.drop(labels=df[df["loudness"] <= -40].index, axis=0, inplace=True)
# print(df["loudness"].describe())

df["loudness"] = pd.cut(x=df["loudness"], bins=[-40, -30, -20, -10, 0])
# print(df["loudness"].unique())
loudness_target = pd.crosstab(index=df["loudness"], columns=df["target"])
# print(loudness_target)
df = pd.get_dummies(data=df, columns=["loudness"])
# print(df.describe(include="all"))
# print(df.columns)
# print(df.shape)

# df["loudness"].plot()
# plt.show()

# mode
# df["mode"].plot()
# plt.show()

# print(df["mode"].describe())
# mode_target = pd.crosstab(index=df["mode"], columns=df["target"])
# print(mode_target)
# print(df.describe())

# speechiness
# df["speechiness"].plot()
# plt.show()

# print(df["speechiness"].describe())
# print(sum(df["speechiness"] > 0.33))
# df.drop(labels=["speechiness"], axis=1, inplace=True)
# print(df.shape)
# df["speechiness"] = pd.cut(x=df["speechiness"], bins=[0, 0.33, 0.66, 1])
# print(df["speechiness"].unique())
# speechiness_target = pd.crosstab(index=df["speechiness"], columns=df["target"])
# print(speechiness_target)
# df = pd.get_dummies(data=df, columns=["speechiness"])

rob_scaler = RobustScaler(quantile_range=(0, 75))
rob_scaled = rob_scaler.fit_transform(df["speechiness"].to_numpy().reshape(-1, 1))
df["speechiness"] = rob_scaled

# print(df.describe())
# print(df.columns)
# print(df.shape)

# df["speechiness"].plot()
# plt.show()

# acousticness
# df["acousticness"].plot()
# plt.show()

# print(len(df["acousticness"].unique()))
# print(df["acousticness"].describe())
# print(sum(df["acousticness"] <= 0.10))
# print(sum(df["acousticness"] >= 0.90))
# acousticness_target = pd.crosstab(index=df["acousticness"], columns=df["target"])

rob_scaler = RobustScaler(quantile_range=(25, 100))
rob_scaled = rob_scaler.fit_transform(df["acousticness"].to_numpy().reshape(-1, 1))
df["acousticness"] = rob_scaled

# print(df["acousticness"].describe())
# print(acousticness_target)

# df["acousticness"].plot()
# plt.show()

# instrumentalness
# df["instrumentalness"].plot()
# plt.show()

# print(df["instrumentalness"].describe())
# print(sum(df["instrumentalness"] <= 0.10))
# print(sum(df["instrumentalness"] >= 0.90))

# rob_scaler = RobustScaler(quantile_range=(0, 75))
# rob_scaled = rob_scaler.fit_transform(df["instrumentalness"].to_numpy().reshape(-1, 1))
# df["instrumentalness"] = rob_scaled

# df["instrumentalness"] = pd.cut(x=df["instrumentalness"], bins=[-1, 0, 0.1, 0.2, 0.4, 0.5, 1])
# print(df["instrumentalness"].unique())
# instrumentalness_target = pd.crosstab(index=df["instrumentalness"], columns=df["target"])
# print(instrumentalness_target)
# df = pd.get_dummies(data=df, columns=["instrumentalness"])
# print(df.describe())
# df.drop(labels=["instrumentalness"], axis=1, inplace=True)

# df["instrumentalness"].plot()
# plt.show()

# sections
# df["sections"].plot()
# plt.show()

# print(df["sections"].unique())
# print(df["sections"].describe())

# bins = [-1, 2]
# for i in range(3, 24):
#     bins.append(i)
# bins.append(145)
# df["sections"] = pd.cut(x=df["sections"], bins=bins)
# print(df["sections"].unique())
# sections_target = pd.crosstab(index=df["sections"], columns=df["target"])
# print(sections_target)
# df = pd.get_dummies(data=df, columns=["sections"])
# print(len(df.columns))
df.drop(labels=["sections"], axis=1, inplace=True)

# print(df.columns)
changed_names = {'loudness_(-40, -30]': 'loudness_3', 'loudness_(-30, -20]': 'loudness_2',
                 'loudness_(-20, -10]': 'loudness_1', 'loudness_(-10, 0]': 'loudness_0'}
df.rename(columns=changed_names, inplace=True)
# print(df.columns)


def score_df(df1):
    X1 = df1.drop(labels=["target"], axis=1)
    y1 = df1["target"]
    scaler = MinMaxScaler()
    drop_feat = ["chorus_hit"]
    scale_feat = ["tempo", "duration_ms", "decade"]
    X2 = X1[scale_feat]
    scaler.fit(X2)
    X2 = pd.DataFrame(scaler.transform(X2), index=X2.index, columns=X2.columns)
    X1.drop(labels=drop_feat, axis=1, inplace=True)
    X1.drop(labels=scale_feat, axis=1, inplace=True)
    X1 = pd.concat([X1, X2], axis=1)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.8, random_state=0)
    for i in range(100, 1100, 100):
        for j in [64]:
            model = RandomForestClassifier(n_estimators=i, max_depth=j, n_jobs=-1, random_state=0)
            model.fit(X1_train, y1_train)
            y1_pred = model.predict(X1_test)
            score = accuracy_score(y1_test, y1_pred)
            print("n_estimators =", i, "max_depth =", j, "Score =", score)


score_df(df)
# At n_estimators=500 and max_depth=64 accuracy=89.42
