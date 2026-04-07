import os
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


def main():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    os.makedirs("outputs", exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=iris.target_names,
        cmap=plt.cm.Blues
    )
    plt.title("Iris Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    joblib.dump(model, "outputs/iris_model.joblib")
    print("Saved confusion matrix to outputs/confusion_matrix.png")
    print("Saved model to outputs/iris_model.joblib")


if __name__ == "__main__":
    main()
