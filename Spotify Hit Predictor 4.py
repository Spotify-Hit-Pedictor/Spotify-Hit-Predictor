import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

decades = ["60", "70", "80", "90", "00", "10"]
df = [pd.read_csv("C:/Users/HP/Documents/Spotify Hit Predictor Dataset/dataset-of-" + i + "s.csv") for i in decades]
for i, decade in enumerate([1960, 1970, 1980, 1990, 2000, 2010]):
    df[i]['decade'] = pd.Series(decade, index=df[i].index)
df = pd.concat(df).reset_index(drop=True)

drop_feature = ["track", "uri", "tempo", "time_signature", "chorus_hit", "sections", "key"]
df.drop(labels=drop_feature, axis=1, inplace=True)

artist = df["artist"].value_counts() >= 10
df["artist"] = df["artist"].apply(lambda x: "Unknown" if not artist.loc[x] else x)

artist_target = pd.crosstab(index=df["artist"], columns=df["target"])
df = pd.get_dummies(data=df, columns=["artist"])

flop_artist_col_0 = "artist_" + artist_target.loc[artist_target[0] == 0].index
flop_artist_col_1 = "artist_" + artist_target.loc[artist_target[0] == 1].index
flop_artist_col_2 = "artist_" + artist_target.loc[artist_target[0] == 2].index
flop_artist_0 = df[flop_artist_col_0].sum(axis=1)
flop_artist_1 = df[flop_artist_col_1].sum(axis=1)
flop_artist_2 = df[flop_artist_col_2].sum(axis=1)

hit_artist_col_0 = "artist_" + artist_target.loc[artist_target[1] == 0].index
hit_artist_col_1 = "artist_" + artist_target.loc[artist_target[1] == 1].index
hit_artist_col_2 = "artist_" + artist_target.loc[artist_target[1] == 2].index
hit_artist_0 = df[hit_artist_col_0].sum(axis=1)
hit_artist_1 = df[hit_artist_col_1].sum(axis=1)
hit_artist_2 = df[hit_artist_col_2].sum(axis=1)

names = {0: "flop_artist_0", 1: "hit_artist_0", 2: "flop_artist_1", 3: "hit_artist_1",
         4: "flop_artist_2", 5: "hit_artist_2"}
df = pd.concat([df, flop_artist_0, hit_artist_0, flop_artist_1, hit_artist_1, flop_artist_2, hit_artist_2], axis=1)

flop_artist_col = list(flop_artist_col_0) + list(flop_artist_col_1) + list(flop_artist_col_2)
hit_artist_col = list(hit_artist_col_0) + list(hit_artist_col_1) + list(hit_artist_col_2)

df.drop(labels=flop_artist_col, axis=1, inplace=True)
df.drop(labels=hit_artist_col, axis=1, inplace=True)
df.rename(columns=names, inplace=True)

col = ['Patti LaBelle', 'Randy Travis', 'Reba McEntire', 'Red Hot Chili Peppers']
for i in range(4):
    col[i] = "artist_" + col[i]
df.drop(labels=col, axis=1, inplace=True)

df.drop(labels=df[df["danceability"] <= 0.10].index, axis=0, inplace=True)
df.drop(labels=df[df["danceability"] >= 0.90].index, axis=0, inplace=True)

scaler = MinMaxScaler()
df["danceability"] = scaler.fit_transform(df["danceability"].to_numpy().reshape(-1, 1))

df.drop(labels=df[df["energy"] <= 0.10].index, axis=0, inplace=True)

scaler = MinMaxScaler()
df["energy"] = scaler.fit_transform(df["energy"].to_numpy().reshape(-1, 1))

scaler = MinMaxScaler()
df["loudness"] = scaler.fit_transform(df["loudness"].to_numpy().reshape(-1, 1))
scaler = MinMaxScaler()
df["duration_ms"] = scaler.fit_transform(df["duration_ms"].to_numpy().reshape(-1, 1))

df = pd.get_dummies(data=df, columns=["decade"])

X = df.drop(labels=["target"], axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=11)
model = RandomForestClassifier(n_estimators=315, max_depth=28, class_weight="balanced", n_jobs=-1, random_state=11)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("n_estimators =", 315, "max_depth =", 32, "Accuracy Score =", score)

# At n_estimators=315 and max_depth=28 the accuracy=90.00 with features=25