# Standard Scikit-Learn Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Interpret & iModels Imports
from interpret.glassbox import ExplainableBoostingClassifier
from imodels import (
    FIGSClassifier,
    RuleFitClassifier,
    SkopeRulesClassifier,
    BoostedRulesClassifier,
    GreedyTreeClassifier,
    TreeGAMClassifier,
)

RANDOM_STATE = 42


SKLEARN_MODELS_SEARCH_SPACE = {
    "Logistic Regression": (
        LogisticRegression(max_iter=100, solver="saga", random_state=RANDOM_STATE),
        {
            "classifier__C": [0.01, 0.1, 1],
            "classifier__class_weight": [None, "balanced"],
            "classifier__l1_ratio":[0, 0.5, 1]
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, 7],
            "classifier__class_weight": [None, "balanced"],
        },
    ),
    "SVM": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__class_weight": [None, "balanced"],
        },
    ),
}


INTERPRETABLE_MODELS_SEARCH_SPACE = {
   
    # "EBM": (
    #     ExplainableBoostingClassifier(random_state=RANDOM_STATE),
    #     {
    #         "classifier__interactions": [0, 5, 10],
    #         "classifier__learning_rate": [0.01, 0.05, 0.1],
    #         "classifier__max_bins": [64, 128],
    #     },
    # ),
  
    "FIGS": (
        FIGSClassifier(random_state=RANDOM_STATE),
        {
            "classifier__max_rules": [10, 20, 30],
            "classifier__max_depth": [2, 3, 5],
            "classifier__max_features": ["sqrt", None],
        },
    ),
   
    "RuleFit": (
        RuleFitClassifier(random_state=RANDOM_STATE),
        {
            "classifier__alpha": [0.001, 0.01, 0.1],
            "classifier__tree_size": [2, 3, 4],
            "classifier__memory_par": [0.01, 0.1],
        },
    ),
  
    # "SkopeRules": (
    #     SkopeRulesClassifier(random_state=RANDOM_STATE),
    #     {
    #         "classifier__n_estimators": [100, 200, 500],
    #         "classifier__max_depth": [2, 3, 4],
    #         "classifier__precision_min": [0.55, 0.7],
    #     },
    # ),
 
    "BoostedRules": (
        BoostedRulesClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        ),
        {
            "classifier__n_estimators": [5, 15, 30],
            "classifier__learning_rate": [0.05, 0.5, 1],
            "classifier__estimator__max_depth": [1, 2],
        },
    ),
  
    "GreedyTree": (
        GreedyTreeClassifier(random_state=RANDOM_STATE),
        {
            "classifier__max_depth": [3, 5, 7],
            "classifier__min_samples_leaf": [5, 10, 20],
            "classifier__class_weight": [None, "balanced"],
        },
    ),
 
    # "TreeGAM": (
    #     TreeGAMClassifier(random_state=RANDOM_STATE),
    #     {
    #         "classifier__n_boosting_rounds": [50, 100, 200],
    #         "classifier__learning_rate": [0.01, 0.05, 0.1],
    #         "classifier__max_leaf_nodes": [2, 3],
    #     },
    # ),
}
