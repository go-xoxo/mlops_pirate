import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC


def isOther(cell, contain_list):
    if cell in contain_list:
        return cell
    return "OTHER"


if __name__ == "__main__":
    # ✅ Load data
    data = pd.read_csv("/data/raw/pirate_data.csv", delimiter=",", index_col=0)

    # ✅ Drop unnecessary columns
    data = data.drop(
        columns=[
            "DATETIME (LOCAL)",
            "DATETIME (UTC)",
            "DATE (UTC)",
            "COUNTRY",
            "MAERSK?",
            "LAT",
            "LONG",
            "TIMEZONE",
            "INCIDENT TYPE",
        ]
    )

    # ✅ Fill missing values
    data = data.fillna(value="UNKNOWN")

    # ✅ Group rare vessel types under 'OTHER'
    vessel_list = (
        data["VESSEL TYPE"]
        .value_counts()
        .index[data["VESSEL TYPE"].value_counts() > 20]
    )
    data["VESSEL TYPE"] = data["VESSEL TYPE"].apply(
        lambda row: isOther(row, vessel_list)
    )

    # ✅ Add time features
    data["MONTH"] = pd.DatetimeIndex(data["DATE (LT)"]).month.astype(str)
    data["WEEKDAY"] = pd.DatetimeIndex(data["DATE (LT)"]).weekday.astype(str)

    # ✅ Drop remaining unnecessary columns
    data = data.drop(columns=["DATE (LT)", "ATTACKS"])

    # ✅ Define features and target
    X = data.drop(["ATTACK SUCCESS"], axis=1)
    y = data["ATTACK SUCCESS"]

    # ✅ Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )

    # ✅ Preprocessing + model
    ohe = preprocessing.OneHotEncoder(handle_unknown="ignore")

    svm = LinearSVC()
    svm_pipe = Pipeline([("one_hot_enc", ohe), ("estimator", svm)])

    svm_score = np.mean(cross_val_score(svm_pipe, X_train, y_train, scoring="f1", cv=4))
    print(f"SVM Score: {svm_score:.3f}")

    svm_rbf = SVC(kernel="rbf")
    svm_rbf_pipe = Pipeline([("one_hot_enc", ohe), ("estimator", svm_rbf)])

    svm_rbf_pipe.fit(X_train, y_train)
    print(svm_rbf_pipe)

    svm_predictions = svm_rbf_pipe.predict(X_test)
