from __future__ import annotations

from pathlib import Path
from typing import Sequence

import kagglehub
import pandas as pd


def load_metabric_data() -> pd.DataFrame:
    """Downloads and loads the METABRIC dataset from Kaggle.

    Fetches the breast cancer gene expression dataset containing RNA and
    mutation data for ~2,000 primary breast cancer samples.

    Returns:
        A pandas DataFrame with METABRIC clinical, mutation, and RNA data.
    """
    print('Fetching METABRIC dataset from Kaggle...')

    dataset_dir = kagglehub.dataset_download(
        'raghadalharbi/breast-cancer-gene-expression-profiles-metabric'
    )

    csv_path = Path(dataset_dir) / 'METABRIC_RNA_Mutation.csv'

    df = pd.read_csv(csv_path, low_memory=False)

    return df




_KNOWN_CLINICAL_FEATURES: frozenset[str] = frozenset({
    'age_at_diagnosis',
    'chemotherapy',
    'cohort',
    'hormone_therapy',
    'lymph_nodes_examined_positive',
    'mutation_count',
    'neoplasm_histologic_grade',
    'nottingham_prognostic_index',
    'radio_therapy',
    'tumor_size',
    'tumor_stage',
})


def count_feature_categories(
    feature_columns: Sequence[str],
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    """Classifies feature columns into clinical, mRNA, and mutation groups.

    Iterates over the provided column names and assigns each to one of three
    categories: known clinical attributes, somatic mutation flags (suffixed
    with '_mut'), or mRNA expression z-scores (everything else).

    Args:
        feature_columns: A sequence of feature column name strings to classify.

    Returns:
        A 4-tuple of:
            - counts: A dict mapping each category label to its column count.
            - clinical_cols: List of matched clinical attribute column names.
            - mrna_cols: List of mRNA z-score column names.
            - mutation_cols: List of mutation flag column names.
    """
    clinical_cols: list[str] = []
    mutation_cols: list[str] = []
    mrna_cols: list[str] = []

    for col in feature_columns:
        if col.endswith('_mut'):
            mutation_cols.append(col)
        elif col in _KNOWN_CLINICAL_FEATURES:
            clinical_cols.append(col)
        else:
            mrna_cols.append(col)

    counts = {
        'Clinical Attributes': len(clinical_cols),
        'm-RNA levels z-score': len(mrna_cols),
        'Mutation': len(mutation_cols),
    }

    for category, count in counts.items():
        print(f'{category}: {count}')

    return counts, clinical_cols, mrna_cols, mutation_cols
