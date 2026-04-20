from __future__ import annotations

from pathlib import Path
from typing import Sequence

import kagglehub
import numpy as np 
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


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
    """Custom Imputer that fills missing values in Histologic Grade and Tumor Stage 
    based on clinical rules (NPI and TNM). Calculates dynamic medians during fit.

    Clinical Rules Applied:
    - Tumor Size (20.0, 50.0): Based on the global TNM cancer staging system, 
      where T1 is <= 20mm, T2 is 20-50mm, and T3 is > 50mm.
    - NPI (3.0, 4.2): Based on the Nottingham Prognostic Index formula 
      [NPI = (0.2 * Size) + Nodes + Grade]. 
          > Since minimum Node score is 1 and Size > 0, an NPI < 3.0 mathematically guarantees a Histological Grade of 1. 
          > Similarly, an NPI < 4.2 makes Grade 3 highly unlikely (minimum NPI for 
            Grade 3 is typically > 4.0), thus we confidently assign Grade 2.
        
    """
    def __init__(self, 
                 npi_low_cutoff=3.0, 
                 npi_mid_cutoff=4.2, 
                 tumor_size_t1_max=20.0, 
                 tumor_size_t2_max=50.0):
        self.npi_low_cutoff = npi_low_cutoff
        self.npi_mid_cutoff = npi_mid_cutoff
        self.tumor_size_t1_max = tumor_size_t1_max
        self.tumor_size_t2_max = tumor_size_t2_max
        
        self.median_grade = None
        self.median_stage = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            # Fallback to 2.0: The global median/mode for the METABRIC cohort
            if 'neoplasm_histologic_grade' in X.columns:
                self.median_grade = X['neoplasm_histologic_grade'].median()
            else:
                self.median_grade = 2.0

            if 'tumor_stage' in X.columns:
                self.median_stage = X['tumor_stage'].median()
            else:
                self.median_stage = 2.0
        return self

    def transform(self, X):
        X_out = X.copy()
        if not isinstance(X_out, pd.DataFrame):
            X_out = pd.DataFrame(X_out)

        def impute_grade(row):
            if not pd.isna(row.get("neoplasm_histologic_grade")): 
                return row["neoplasm_histologic_grade"]
            
            npi = row.get("nottingham_prognostic_index")
            if pd.isna(npi): return self.median_grade 
            
            if npi < self.npi_low_cutoff: return 1.0
            if npi < self.npi_mid_cutoff: return 2.0
            return 3.0

        def impute_stage(row):
            if not pd.isna(row.get("tumor_stage")): 
                return row["tumor_stage"]
            
            size = row.get("tumor_size")
            ln = row.get("lymph_nodes_examined_positive")
            if pd.isna(size) or pd.isna(ln): return self.median_stage 
            
            if size > self.tumor_size_t2_max or ln >= 4: return 3.0
            if size <= self.tumor_size_t1_max and ln == 0: return 1.0
            return 2.0

        if "neoplasm_histologic_grade" in X_out.columns:
            X_out["neoplasm_histologic_grade"] = X_out.apply(impute_grade, axis=1)
        if "tumor_stage" in X_out.columns:
            X_out["tumor_stage"] = X_out.apply(impute_stage, axis=1)

        return X_out


class DiscordantSignatureAdder(BaseEstimator, TransformerMixin):
    """Creates the combined molecular score of the 3 genes."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if not isinstance(X_out, pd.DataFrame):
            X_out = pd.DataFrame(X_out)

        required_genes = ['stat5a', 'mmp11', 'col22a1']
        if all(gene in X_out.columns for gene in required_genes):
            # Score Formula: (MMP11 + COL22A1) - STAT5A
            X_out['discordant_molecular_score'] = (
                X_out['mmp11'] + 
                X_out['col22a1'] - 
                X_out['stat5a']
            )
        return X_out