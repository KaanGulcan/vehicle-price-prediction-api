import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error

# DATA READING

df = pd.read_csv("../data/data.csv")

# DATA ANALYISIS

# print(df.head(10))
# print(df.isnull().sum())
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.columns)

# FEATURE ENGINEERING

df["brand"] = df["name"].apply(lambda x: x.split(" ")[0])
df["model"] = df["name"].apply(lambda x: " ".join(x)[1:])

df["engine"] = df["engine"].str.replace(" CC","").astype(float)

# DATA CLEANING

df.drop(["name","mileage","max_power","torque"], axis=1, inplace=True)
df.dropna(subset=["engine","seats"], axis=0, inplace=True)

# FEATURE SELECTION

X = df.drop(["selling_price"], axis=1)
y = df["selling_price"]

# TRAIN/TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COLUMNS

num_columns = ["engine","km_driven","seats","year"]
cat_columns = ["fuel","seller_type","transmission","owner","brand","model"]

# PREPROCESSOR

preprocessor = ColumnTransformer([
    ("num",StandardScaler(),num_columns),
    ("cat",OneHotEncoder(handle_unknown="ignore"),cat_columns)
])

# PIPELINE

pipeline = Pipeline([
    ("preprocessor",preprocessor),
    ("model",LinearRegression())
])

# TRAIN

y_train_log = np.log1p(y_train)

pipeline.fit(X_train,y_train_log)

# PREDICTION

y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)

# EVALUATION

print(f"MAE: {mean_absolute_error(y_test,y_pred)}")
print(f"RMSE: {root_mean_squared_error(y_test,y_pred)}")
print(f"R2: {r2_score(y_test,y_pred)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test,y_pred)}")

# JOBLIB

joblib.dump(pipeline,"../models/linRegr_model.pkl")

print("Model Successfuly Saved!")