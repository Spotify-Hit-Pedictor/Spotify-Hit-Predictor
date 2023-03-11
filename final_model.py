import pandas as pd
import xgboost as xb
import pickle as pk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

decades = ["60", "70", "80", "90", "00", "10"]
df = [pd.read_csv("./datasets/dataset-of-" + i + "s.csv") for i in decades]
for i, decade in enumerate([1960, 1970, 1980, 1990, 2000, 2010]):
    df[i]["decade"] = pd.Series(decade, index=df[i].index)
df = pd.concat(df)
df.reset_index(drop=True, inplace=True)

df = df.drop(["track", "artist", "uri", "key", "chorus_hit"], axis=1)

scaler = StandardScaler()
col = ["tempo", "decade", "duration_ms", "loudness", "sections", "time_signature"]
scaler.fit(df[col])
df[col] = scaler.transform(df[col])

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values

del df_train['target']
del df_val['target']
del df_test['target']

train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

# rf = RandomForestClassifier(n_estimators=500, max_depth=32, class_weight="balanced", n_jobs=-1, random_state=11)
# rf.fit(X_train, y_train)
#
# y_pred = rf.predict_proba(X_val)[:, 1]
# score = roc_auc_score(y_val, y_pred)
# print("n_estimators =", 500, "max_depth =", 32, "Accuracy Score =", score)

# At n_estimators=500 and max_depth=32 the accuracy=88.51 with features=14

# xgb = XGBClassifier()
# xgb.fit(X_train, y_train)
#
# y_pred = xgb.predict_proba(X_val)[:, 1]
# score = roc_auc_score(y_val, y_pred)
# print(score)

features = dv.get_feature_names_out()
dtrain = xb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xb.DMatrix(X_val, label=y_val, feature_names=features)

xgb_params = {
    'eta': 0.08,
    'max_depth': 100,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

watchlist = [(dtrain, "train"), (dval, "val")]
model = xb.train(xgb_params, dtrain, num_boost_round=305, verbose_eval=5)

y_pred = model.predict(dval)
score = roc_auc_score(y_val, y_pred)
# print(score)

file_name = 'final_model.bin'
pk.dump(model, open(file_name, 'wb'))