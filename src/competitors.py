# -*- coding: utf-8 -*-
"""
Logic for benchmarking against competitive models and published gene signatures.
Paper: Rezaeian et al. (2017)
Link: https://doi.org/10.1101/105403
"""

import matplotlib.pyplot as plt
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
    """Returns the exact list of 39 genes identified in the Rezaeian study."""
    return [
        'bbc3', 'cdkn2a', 'wfdc2', 'hes2', 'hsd3b7', 'rps6kb1', 'e2f6', 'srd5a3',
        'mmp25', 'mmp21', 'dnah5', 'sf3b1', 'chek2', 'sik1', 'bmpr2', 'hey1',
        'mmp27', 'stmn2', 'mapk3', 'pdgfa', 'taf4b', 'ctbp1', 'igf1r', 'notch2',
        'arid2', 'arid1a', 'spry2', 'cdkn2c', 'gata3', 'spen', 'ccnd2', 'smad2',
        'map3k5', 'prkg1', 'terc', 'hes6', 'nrarp', 'agtr2', 'pde4dip'
    ]

def map_competitor_features(available_columns, competitor_list):
    """
    Maps the gene list to exactly one column per gene.
    Prioritizes mRNA expression over Mutation status to reach an exact count.
    """
    mapped = []
    available_set = set(available_columns)

    print("Available columns:", available_columns[0:5])
    print("Competitors:", competitor_list[0:5])

    for gene in competitor_list:
        # Try to find the mRNA column
        if gene in available_set:
            print("Gene added:", gene)
            mapped.append(gene)
        # If not found, try the Mutation column
        elif f"{gene}_mut" in available_set:
            print("Gene addded (mutation):", gene)
            mapped.append(f"{gene}_mut")

    print(f"Strict Mapping: Found exactly {len(mapped)} features for the {len(competitor_list)} target genes.")
    return mapped

def export_results(
    model,
    X_test,
    y_test,
    threshold: float | None = None,
    suffix: str = "",
    save_file: bool = False,
):
    """Generate and display a confusion matrix for a binary classifier.

    The function evaluates a classification model on test data, selects an
    operating threshold (either provided or PR-curve–optimal), and plots the
    corresponding confusion matrix.

    Args:
        model: Trained model implementing `predict_proba`.
        X_test: Feature matrix for evaluation.
        y_test: Ground-truth labels.
        threshold: Classification threshold. If None, the threshold that
            maximizes F1-score on the precision-recall curve is used.
        suffix: Descriptive label added to plots and filenames.
        save_file: Whether to save the confusion matrix plot as a PNG file.

    Returns:
        np.ndarray: Binary predictions generated using the selected threshold.

    Notes:
        - Average Precision (AP) is computed using predicted probabilities.
        - When `threshold` is None, the PR-optimal threshold maximizes F1-score.
        - Confusion matrix is displayed using matplotlib.
    """
    probs = model.predict_proba(X_test)[:, 1]
    average_precision = average_precision_score(y_test, probs)

    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = (2 * precisions * recalls) / (
            precisions + recalls + 1e-10
        )

        best_index = int(np.argmax(f1_scores))
        threshold_index = min(best_index, len(thresholds) - 1)
        used_threshold = thresholds[threshold_index]
        f1_value = f1_scores[best_index]
    else:
        used_threshold = threshold
        f1_value = f1_score(
            y_test, (probs >= used_threshold).astype(int)
        )

    print(
        f"[{suffix}] "
        f"Threshold: {used_threshold:.3f} | "
        f"AP: {average_precision:.3f} | "
        f"F1: {f1_value:.3f}"
    )

    predictions = (probs >= used_threshold).astype(int)
    confusion = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(confusion_matrix=confusion)
    display.plot(cmap="Blues")

    plt.title(
        f"Confusion Matrix {suffix}\n"
        f"Threshold: {used_threshold:.3f}"
    )

    if save_file:
        filename = f"benchmark_cm_{suffix.replace(' ', '_')}.png"
        plt.savefig(filename)

    plt.show()
    return predictions
