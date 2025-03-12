import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as ss

data = pd.read_csv("./data/Student Depression Dataset.csv")


# Data cleaning
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


data = preprocess_data(data)
print(data.isnull().sum())

data.info()
