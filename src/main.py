import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as ss

data = pd.read_csv("./data/Student Depression Dataset.csv")


def preprocess_data(df):
    """Take the data and preprocess it by removing non-pertinent columns and encoding the values so that it's numerical"""
    # Removing non-pertinent columns
    df.drop(
        columns=[
            "id",
            "Gender",
            "Degree",
            "Age",
            "City",
            "Profession",
            "Job Satisfaction",
        ],
        inplace=True,
    )
    # Filling the missing values
    df["Financial Stress"] = df["Financial Stress"].fillna(
        df["Financial Stress"].mean()
    )
    # Encoding data
    numerical_columns = [
        "Academic Pressure",
        "Work Pressure",
        "CGPA",
        "Study Satisfaction",
        "Work/Study Hours",
        "Financial Stress",
    ]
    ordinal_columns = {
        "Sleep Duration": [
            "Less than 5 hours",
            "5-6 hours",
            "7-8 hours",
            "More than 8 hours",
            "Others",
        ],
        "Have you ever had suicidal thoughts ?": ["No", "Yes"],
        "Family History of Mental Illness": ["No", "Yes"],
        "Dietary Habits": ["Unhealthy", "Moderate", "Healthy", "Others"],
    }
    ordinal_encoder = OrdinalEncoder(
        categories=[ordinal_columns[col] for col in ordinal_columns]
    )
    df[list(ordinal_columns.keys())] = ordinal_encoder.fit_transform(
        df[list(ordinal_columns.keys())]
    )

    return df


def split_data(df):
    """Spliting data between training and testing set"""

    X = df.drop(columns=["Depression"])
    Y = df["Depression"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    return X_train, Y_train, X_test, Y_test


def training(clf, X_train, Y_train, X_test, Y_test):

    # Training
    clf.fit(X_train, Y_train)
    # Testing
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    return clf, accuracy


data = preprocess_data(data)
# print(data.isnull().sum())

# data.info()
X_train, Y_train, X_test, Y_test = split_data(data)
DT = DecisionTreeClassifier()
RF = RandomForestClassifier(n_estimators=100, random_state=42)
NN = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
models = {"Decision Tree": DT, "Random Forest": RF, "Neural Network": NN}
for key, clf in models.items():
    clf, acurracy = training(clf, X_train, Y_train, X_test, Y_test)
    print(f"{key} have {acurracy} score")
