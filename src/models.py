from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

RANDOM_STATE=42

MODELS_WITH_SEARCH_SPACE = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
        {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, 7],
            "classifier__class_weight": [None, "balanced"]
        }
    ),
    "SVM": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__class_weight": [None, "balanced"]
        }
    ),
}