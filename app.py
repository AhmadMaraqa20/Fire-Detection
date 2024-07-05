import streamlit as st
import pandas as pd

!pip install joblib 

import joblib

# Load the trained model
model = joblib.load("DTC_model.pkl")


import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf


file_path = r"smoke_detection_iot.csv"
data = pd.read_csv(file_path)
analysis_data = data.drop(data.columns.tolist()[:2], axis=1)


def column_add(X):
    X["PM"] = X["PM1.0"] + X["PM2.5"]
    X["NC"] = X["NC0.5"] + X["NC1.0"] + X["NC2.5"]
    return X


def remove_columns(X):
    X.drop(
        ["index_d", "UTC", "PM1.0", "PM2.5", "NC0.5", "NC1.0", "NC2.5", "CNT"],
        axis=1,
        inplace=True,
    )
    return X


combine_transformer = FunctionTransformer(column_add)
remove_transformer = FunctionTransformer(remove_columns)

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
for train, test in sss.split(data, data["Fire Alarm"]):
    train_data = data.iloc[train]
    test_data = data.iloc[test]
train_X = train_data.drop("Fire Alarm", axis=1)
train_Y = train_data["Fire Alarm"].copy()
test_X = test_data.drop("Fire Alarm", axis=1)
test_Y = test_data["Fire Alarm"].copy()

numerical_pipeline = Pipeline(
    [
        ("CombineFeautres", combine_transformer),
        ("RemoveFeatures", remove_transformer),
        ("Imputer", SimpleImputer(strategy="median")),
        ("Scaler", StandardScaler()),
        (
            "Polynomial",
            PolynomialFeatures(degree=1),
        ),  # Perform better without raising the degree of the features
    ]
)
log_pipeline = Pipeline(
    [
        ("Imputer", SimpleImputer(strategy="median")),
        ("Log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
        ("Scaler", StandardScaler()),
    ]
)

num_scale_columns = [x for x in list(data) if x not in ["Pressure[hPa]", "Fire Alarm"]]

preprocessing = ColumnTransformer(
    [
        ("log", log_pipeline, ["Pressure[hPa]"]),
        ("num_scale", numerical_pipeline, num_scale_columns),
    ],
    remainder="drop",
)

train_X = preprocessing.fit_transform(train_X)
test_X = preprocessing.transform(test_X)

train_X = np.delete(train_X, 1, axis=1)
test_X = np.delete(test_X, 1, axis=1)


# Define the columns expected by the model
columns = [
    "Temperature[C]",
    "Humidity[%]",
    "TVOC[ppb]",
    "eCO2[ppm]",
    "Raw H2",
    "Raw Ethanol",
    "Pressure[hPa]",
    "PM1.0",
    "PM2.5",
    "NC0.5",
    "NC1.0",
    "NC2.5",
    "CNT",
]

# Create a Streamlit app
st.title("Fire Detection App")

# Create a table for user input
st.header("Input Data")
input_data = {}
input_data["index_d"] = 10
input_data["UTC"] = 10
for col in columns:
    input_data[col] = st.number_input(col, value=5.000)

# Convert the input data into a DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the input data
input_preprocessed = preprocessing.transform(input_df)

# Convert preprocessed data to DataFrame and drop the extra polynomial feature column if necessary
input_preprocessed_df = pd.DataFrame(
    input_preprocessed, columns=[f"F{i}" for i in range(input_preprocessed.shape[1])]
)
if "F1" in input_preprocessed_df.columns:
    input_preprocessed_df.drop("F1", axis=1, inplace=True)

# Display the input DataFrame
st.subheader("Input DataFrame")
input_df.index = ["data"]
st.write(input_df.iloc[:, 2:])


# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_preprocessed_df)
    result = (
        "Fire Detected" if prediction[0] >= 0.5 else "No Fire"
    )  # Adjust threshold as necessary
    st.subheader("Prediction")
    st.write(result)


# cd C:\Users\HP\OneDrive\Desktop\work\AI portfolio\AI
# streamlit run app.py
