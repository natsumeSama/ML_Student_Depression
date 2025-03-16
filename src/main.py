import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

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
    """Train and test the classifier clf with the training and testing set in the parameters"""
    # Training
    clf.fit(X_train, Y_train)
    # Testing
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    return clf, accuracy


def accuracy(models, accuracy_scores):
    """Calculate the Accuracy Scores of all the models in models"""
    for key, clf in models.items():
        clf, acurracy = training(clf, X_train, Y_train, X_test, Y_test)
        accuracy_scores.append(acurracy)
        print(f"{key} have {acurracy} score")


def plot(models, accuracy_scores):
    """Draw a Bar chart comparing each models based on it's accuracy score"""
    plt.figure(figsize=(10, 8))
    plt.bar(
        models.keys(),
        accuracy_scores,
        color=["skyblue", "lightgreen", "salmon", "orange"],
    )

    plt.title("Accuracy Score Comparaison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, score in enumerate(accuracy_scores):
        plt.text(i, score + 0.01, f"{score:.2f}", ha="center")

    plt.show()


data = preprocess_data(data)

X_train, Y_train, X_test, Y_test = split_data(data)
# Creating classifiers
DT = DecisionTreeClassifier(criterion="entropy", random_state=42)
DTP = DecisionTreeClassifier(max_depth=7, random_state=42)
RF = RandomForestClassifier(n_estimators=100, random_state=42)
NN = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
accuracy_scores = []
models = {
    "Decision Tree": DT,
    "Decision Tree Pruned": DTP,
    "Random Forest": RF,
    "Neural Network": NN,
}

accuracy(models, accuracy_scores)
plot(models, accuracy_scores)
