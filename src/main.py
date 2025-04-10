import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("./data/Student Depression Dataset.csv")


def preprocess_data(df):
    """Clean and preprocess the dataset by removing irrelevant columns,
    handling missing values, and encoding categorical variables."""

    # Remove irrelevant columns
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

    # Fill missing values with the mean
    df["Financial Stress"] = df["Financial Stress"].fillna(
        df["Financial Stress"].mean()
    )

    # Encode categorical variables using ordinal encoding
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
    """Split the dataset into training and testing sets."""

    X = df.drop(columns=["Depression"])
    Y = df["Depression"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    return X_train, Y_train, X_test, Y_test


def training(clf, X_train, Y_train, X_test, Y_test):
    """Train the classifier and evaluate its performance on the test set."""

    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)
    return clf, accuracy, cm


def accuracy(models, accuracy_scores, confusion_matrixs):
    """Train and evaluate all classifiers, collecting their accuracy and confusion matrices."""

    for key, clf in models.items():
        clf, acurracy, cm = training(clf, X_train, Y_train, X_test, Y_test)
        accuracy_scores.append(acurracy)
        confusion_matrixs.append(cm)
        print(f"{key} has an accuracy score of {acurracy:.2f}")


def plot_accuracy_bar(models, accuracy_scores):
    """Display a bar plot comparing the accuracy scores of different models."""

    plt.figure(figsize=(10, 6))
    plt.bar(
        models.keys(),
        accuracy_scores,
        color=["skyblue", "lightgreen", "salmon", "orange"],
    )
    plt.title("Accuracy Score Comparison", fontsize=16)
    plt.xlabel("Models")
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, score in enumerate(accuracy_scores):
        plt.text(i, score + 0.01, f"{score:.2f}", ha="center", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(models, confusion_matrixs):
    """Display one confusion matrix per model, with aligned labels and titles."""

    import matplotlib.patheffects as path_effects

    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(5.5 * num_models, 5))

    if num_models == 1:
        axes = [axes]  # Ensure axes is iterable even with a single model

    for ax, (model_name, cm) in zip(axes, zip(models.keys(), confusion_matrixs)):
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(model_name, fontsize=14, pad=50)

        # Remove tick marks
        ax.set_xticks([])
        ax.set_yticks([])

        # Display cell values
        thresh = cm.max() / 2
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(
                    k,
                    j,
                    format(cm[j, k], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[j, k] > thresh else "black",
                    fontsize=13,
                    path_effects=[
                        path_effects.withStroke(linewidth=1.5, foreground="black")
                    ],
                )

        # Column labels (top)
        col_labels = ["No Depression", "Depression"]
        for idx, label in enumerate(col_labels):
            ax.text(
                idx,
                -0.9,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Row labels (left)
        row_labels = ["No Depression", "Depression"]
        for idx, label in enumerate(row_labels):
            ax.text(
                -0.9,
                idx,
                label,
                ha="right",
                va="center",
                fontsize=10,
            )

        # "Predicted" label centered below the model title
        ax.text(
            0.5,
            -1.7,
            "Predicted",
            ha="center",
            va="bottom",
            transform=ax.transData,
            fontsize=12,
            fontstyle="italic",
        )

        # "True" label on the left, vertically centered
        ax.text(
            -1.8,
            0.5,
            "True",
            ha="center",
            va="center",
            rotation=90,
            transform=ax.transData,
            fontsize=12,
            fontstyle="italic",
        )

    plt.tight_layout()
    plt.show()


def plot(models, accuracy_scores, confusion_matrixs):
    """Run both the accuracy bar plot and the confusion matrix display."""
    plot_accuracy_bar(models, accuracy_scores)
    plot_confusion_matrices(models, confusion_matrixs)


# Preprocessing, training and plotting pipeline
data = preprocess_data(data)

X_train, Y_train, X_test, Y_test = split_data(data)

# Create classifiers
DT = DecisionTreeClassifier(criterion="entropy", random_state=42)
DTP = DecisionTreeClassifier(max_depth=7, random_state=42)
RF = RandomForestClassifier(n_estimators=100, random_state=42)
NN = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

accuracy_scores = []
confusion_matrixs = []

models = {
    "Decision Tree": DT,
    "Decision Tree Pruned": DTP,
    "Random Forest": RF,
    "Neural Network": NN,
}

# Evaluate and visualize
accuracy(models, accuracy_scores, confusion_matrixs)
plot(models, accuracy_scores, confusion_matrixs)
