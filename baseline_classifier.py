import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
DATA = "data/nasa/data_2016.csv"


def classifier_train():
    df = pd.read_csv(DATA)
    df = df.iloc[:, 2:]
    X = df.iloc[:, 1:54]
    Y = df.iloc[:, 55]
    xtrain, xvalid, ytrain, yvalid = train_test_split(X, Y, test_size=0.2)
    classifier = xgboost.XGBClassifier()

    classifier.fit(
        xtrain, ytrain,
    )
    predictions = classifier.predict(xvalid)
    accuracy = accuracy_score(yvalid, predictions)
    print(accuracy)
    return classifier


if __name__ == "__main__":
    classifier_train()
