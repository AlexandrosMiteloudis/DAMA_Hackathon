# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV


def get_rezaeian_features():
    """Returns the exact list of 39 features identified in the Rezaeian study."""
    return [
        'bbc3', 'cdkn2a', 'wfdc2', 'hes2', 'hsd3b7', 'rps6kb1', 'e2f6', 'srd5a3',
        'mmp25', 'mmp21', 'dnah5', 'sf3b1', 'chek2', 'sik1', 'bmpr2', 'hey1',
        'mmp27', 'stmn2', 'mapk3', 'pdgfa', 'taf4b', 'ctbp1', 'igf1r', 'notch2',
        'arid2', 'arid1a', 'spry2', 'cdkn2c', 'gata3', 'spen', 'ccnd2', 'smad2',
        'map3k5', 'prkg1', 'terc', 'hes6', 'nrarp', 'agtr2', 'pde4dip'
    ]

def map_competitor_features(available_columns, competitor_list):
    """Maps the published gene list to the current dataset's column names."""
    mapped = []
    for col in available_columns:
        base_gene = col.replace('_mut', '').lower()
        if base_gene in competitor_list:
            mapped.append(col)

    print(f"Mapped {len(mapped)} features out of the {len(competitor_list)} features selected by the competitor.")
    return mapped

def export_benchmark_matrices(model, X_test, y_test, threshold=None, suffix="", save_file=False):
    """
    Based on the given classification threshold, it does adapt the decisions
    of the provided model, and produces the corresponding confusion matrix.
    Note: It does work only with binary classification problems!

    If threshold is None, it calculates the PR-Optimal Threshold (Max F1).
    """
    probs = model.predict_proba(X_test)[:, 1]
    ap_score = average_precision_score(y_test, probs)

    # Logic to handle Tuned vs Untuned
    if threshold is None:
        # Tuning logic: Find the best threshold using Precision-Recall Curve
        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        used_threshold = thresholds[min(best_idx, len(thresholds)-1)]
        mode_label = "Tuned"
    else:
        # Use the fixed threshold provided (e.g., 0.5)
        used_threshold = threshold
        mode_label = "Fixed"

    print(f"[{suffix}] Mode: {mode_label} | Threshold: {used_threshold:.3f} | AP: {ap_score:.3f}")

    # Generate Predictions and Matrix
    preds = (probs >= used_threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix ({suffix})\nThreshold: {used_threshold:.3f} | AP: {ap_score:.3f}")

    if save_file:
        plt.savefig(f"benchmark_cm_{suffix.replace(' ', '_')}.png")
    plt.show()
