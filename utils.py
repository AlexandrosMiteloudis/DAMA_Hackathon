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
    'cohort',
    'lymph_nodes_examined_positive',
    'mutation_count',
    'neoplasm_histologic_grade',
    'nottingham_prognostic_index',
    'tumor_size',
    'tumor_stage',
    'cellularity',
    'er_status',
    'pr_status',
    'her2_status'
})


def count_feature_categories(
    feature_columns: Sequence[str],
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    """Classifies feature columns into clinical, mRNA, and mutation groups.

    Iterates over the provided column names and assigns each to one of three
    categories: known clinical attributes, somatic mutation flags (suffixed
    with '_mut'), or mRNA expression z-scores (everything else).

    It is assumed that the provided column names are these of the METABRIC dataset

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

    return counts, clinical_cols, mrna_cols, mutation_cols



class SmartClinicalImputer(BaseEstimator, TransformerMixin):
    """ Custom Imputer that fills missing values in Histologic Grade and Tumor Stage 
    based on clinical rules (NPI and TNM).
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        if not isinstance(X_out, pd.DataFrame):
            X_out = pd.DataFrame(X_out)
            
        def impute_grade(row):
            if not pd.isna(row.get("neoplasm_histologic_grade")): 
                return row["neoplasm_histologic_grade"]
            npi = row.get("nottingham_prognostic_index")
            if pd.isna(npi): return 2.0
            if npi < 3.0: return 1.0
            if npi < 4.2: return 2.0
            return 3.0
            
        def impute_stage(row):
            if not pd.isna(row.get("tumor_stage")): 
                return row["tumor_stage"]
            size = row.get("tumor_size")
            ln = row.get("lymph_nodes_examined_positive")
            if pd.isna(size) or pd.isna(ln): return 2.0
            if size > 50 or ln >= 4: return 3.0
            if size <= 20 and ln == 0: return 1.0
            return 2.0
        
        if "neoplasm_histologic_grade" in X_out.columns:
            X_out["neoplasm_histologic_grade"] = X_out.apply(impute_grade, axis=1)
        if "tumor_stage" in X_out.columns:
            X_out["tumor_stage"] = X_out.apply(impute_stage, axis=1)
            
        return X_out

class DiscordantSignatureAdder(BaseEstimator, TransformerMixin):
    """ Creates the combined molecular score of the 3 genes. 
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        if not isinstance(X_out, pd.DataFrame):
            X_out = pd.DataFrame(X_out)
            
        required_genes = ['stat5a', 'mmp11', 'col22a1']
        if all(gene in X_out.columns for gene in required_genes):
            X_out['discordant_molecular_score'] = (
                (-1 * X_out['stat5a']) + 
                (1 * X_out['mmp11']) + 
                (1 * X_out['col22a1'])
            )
        return X_out
