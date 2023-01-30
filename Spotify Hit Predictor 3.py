# Importing libraries
import pandas as pd

# Importing functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Setting display configurations
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Fetching data
! wget https://raw.githubusercontent.com/Spotify-Hit-Pedictor/Spotify-Hit-Predictor/main/archive/dataset-of-60s.csv 
! wget https://raw.githubusercontent.com/Spotify-Hit-Pedictor/Spotify-Hit-Predictor/main/archive/dataset-of-70s.csv 
! wget https://raw.githubusercontent.com/Spotify-Hit-Pedictor/Spotify-Hit-Predictor/main/archive/dataset-of-80s.csv 
! wget https://raw.githubusercontent.com/Spotify-Hit-Pedictor/Spotify-Hit-Predictor/main/archive/dataset-of-90s.csv 
! wget https://raw.githubusercontent.com/Spotify-Hit-Pedictor/Spotify-Hit-Predictor/main/archive/dataset-of-00s.csv 
! wget https://raw.githubusercontent.com/Spotify-Hit-Pedictor/Spotify-Hit-Predictor/main/archive/dataset-of-10s.csv

# Loading dataset
decades = ["60", "70", "80", "90", "00", "10"]
df = [pd.read_csv("dataset-of-" + i + "s.csv") for i in decades]
for i, decade in enumerate([1960, 1970, 1980, 1990, 2000, 2010]):
    df[i]['decade'] = pd.Series(decade, index=df[i].index)
df = pd.concat(df).reset_index(drop=True)

# Track feature
df.drop(labels=["track"], axis=1, inplace=True)

# Uri feature
df.drop(labels=["uri"], axis=1, inplace=True)

# Artist feature
artist = df["artist"].value_counts() > 10
df["artist"] = df["artist"].apply(lambda x: "Unknown" if not artist.loc[x] else x)

artist_target = pd.crosstab(index=df["artist"], columns=df["target"])
df = pd.get_dummies(data=df, columns=["artist"])

flop_artist_col = "artist_" + artist_target.loc[artist_target[0] == 0].index
flop_artist = df[flop_artist_col].sum(axis=1)

hit_artist_col = "artist_" + artist_target.loc[artist_target[1] == 0].index
hit_artist = df[hit_artist_col].sum(axis=1)

names = {0: "flop_artist", 1: "hit_artist"}
df = pd.concat([df, flop_artist, hit_artist], axis=1)

df.drop(labels=flop_artist_col, axis=1, inplace=True)
df.drop(labels=hit_artist_col, axis=1, inplace=True)
df.rename(columns=names, inplace=True)

# Danceability feature
df.drop(labels=df[df["danceability"] <= 0.10].index, axis=0, inplace=True)
df.drop(labels=df[df["danceability"] >= 0.90].index, axis=0, inplace=True)

scaler = MinMaxScaler()
df["danceability"] = scaler.fit_transform(df["danceability"].to_numpy().reshape(-1, 1))

# Energy feature
df.drop(labels=df[df["energy"] <= 0.10].index, axis=0, inplace=True)

scaler = MinMaxScaler()
df["energy"] = scaler.fit_transform(df["energy"].to_numpy().reshape(-1, 1))

# Key feature
df = pd.get_dummies(data=df, columns=["key"])

# Loudness feature
df.drop(labels=df[df["loudness"] >= 0].index, axis=0, inplace=True)
df.drop(labels=df[df["loudness"] <= -40].index, axis=0, inplace=True)

labels = ['3', '2', '1', '0']
df["loudness"] = pd.cut(x=df["loudness"], bins=[-40, -30, -20, -10, 0], labels=labels)
df = pd.get_dummies(data=df, columns=["loudness"])

# Tempo feature
df.drop(labels=["tempo"], axis=1, inplace=True)

# Duration_ms feature
df["duration_ms"] = scaler.fit_transform(df["duration_ms"].to_numpy().reshape(-1, 1))

# Time_signature feature
df.drop(labels=["time_signature"], axis=1, inplace=True)

# Chorus_hit feature
df.drop(labels=["chorus_hit"], axis=1, inplace=True)

# Sections feature
df.drop(labels=["sections"], axis=1, inplace=True)

# Decade feature
df = pd.get_dummies(data=df, columns=["decade"])

# Random Forest Classifier Model
X = df.drop(labels=["target"], axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
model = RandomForestClassifier(n_estimators=1400, max_depth=64, n_jobs=-1, class_weight="balanced_subsample", random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("n_estimators =", 1400, "max_depth =", 64, "Score =", score)

# At n_estimators=1400 and max_depth=64 accuracy=89.50
