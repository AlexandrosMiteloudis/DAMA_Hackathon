from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import kagglehub
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .mappings import KNOWN_CLINICAL_FEATURES


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
        elif col in KNOWN_CLINICAL_FEATURES:
            clinical_cols.append(col)
        else:
            mrna_cols.append(col)

    counts = {
        'Clinical Attributes': len(clinical_cols),
        'm-RNA levels z-score': len(mrna_cols),
        'Mutation': len(mutation_cols),
    }

    return counts, clinical_cols, mrna_cols, mutation_cols

def build_metabric_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, set[str]]:
    """Prepares features and target variable for the METABRIC dataset.

    Filters for the Luminal A subtype, separates target variables
    to prevent data leakage, encodes specific categorical columns,
    and applies one-hot encoding to any remaining text variables.
    Finally, it returns the processed feature matrix, target series, and a set of any dropped columns for reference.
    The dropped columns are the non-numerical ones.

    Args:
        df: The raw METABRIC pandas DataFrame.

    Returns:
        A tuple containing:
            - X: The processed feature matrix (pandas DataFrame) ready for ML.
            - y: The target mortality variable (pandas Series).
            - dropped_columns: A set of column names that were dropped at the last stage as non numerical.
    """
    df = df.copy()

    # Subtype filtering
    df = df[df['pam50_+_claudin-low_subtype'] == 'LumA'].copy()

    # Target encoding
    def encode_mortality(value) -> int:
        if str(value).strip().lower() == "died of disease":
            return 1
        return 0

    df['target_mortality'] = df['death_from_cancer'].apply(encode_mortality)

    # Leakage removal
    leakage_cols = [
        'patient_id', 'overall_survival_months', 'overall_survival',
        'death_from_cancer', 'target_mortality', 'chemotherapy',
        'hormone_therapy', 'radio_therapy', 'type_of_breast_surgery'
    ]
    X = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    y = df['target_mortality']

    # Encoding specific known categorical features
    for col in ["er_status", "her2_status", "pr_status"]:
        if col in X.columns:
            X[col] = (X[col] == "Positive").astype(int)

    if "cellularity" in X.columns:
        X["cellularity"] = X["cellularity"].map({"Low": 0, "Moderate": 1, "High": 2})

    # Mutation binarization
    mutation_cols = [c for c in X.columns if c.endswith("_mut")]
    for col in mutation_cols:
        def encode_mutation(value):
            if pd.isna(value) or str(value).strip() == "0":
                return 0
            return 1
        X[col] = X[col].apply(encode_mutation)

    # Categorical Encoding for ML Models
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    before_reduction = X.columns
    # Keep numeric only just to be safe
    X = X.select_dtypes(include=['number'])
    
    dropped_cols = set(before_reduction) - set(X.columns)
    return X, y, dropped_cols

class SmartClinicalImputer(BaseEstimator, TransformerMixin):
    """Imputes missing clinical data (Grade and Stage) using medical heuristics.

    Leverages Tumor Size and Lymph Nodes to infer missing values. Falls back
    to the population median if clinical heuristics cannot be applied.
    """
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            if 'neoplasm_histologic_grade' in X.columns:
                self.median_grade = X['neoplasm_histologic_grade'].median()
            else:
                raise ValueError("Missing neoplasm_histologic_grade column")

            if 'tumor_stage' in X.columns:
                self.median_stage = X['tumor_stage'].median()
            else:
                raise ValueError("Missing tumor_stage column")
        return self

    def transform(self, X):
        X_out = X.copy()
        
        def impute_grade(row):
            if not pd.isna(row.get("neoplasm_histologic_grade")): 
                return row["neoplasm_histologic_grade"]
            npi = row.get("nottingham_prognostic_index")
            if pd.isna(npi): 
                return self.median_grade 
            if npi < 3.0: 
                return 1.0
            if npi < 4.2: 
                return 2.0
            return 3.0

        def impute_stage(row):
            if not pd.isna(row.get("tumor_stage")): 
                return row["tumor_stage"]
            size = row.get("tumor_size")
            ln = row.get("lymph_nodes_examined_positive")
            if (pd.isna(size) or pd.isna(ln)):
                return self.median_stage 
            if (size > 50.0 or ln >= 4):
                return 3.0
            if (size <= 20.0 and ln == 0): 
                return 1.0
            return 2.0

        if "neoplasm_histologic_grade" in X_out.columns:
            X_out["neoplasm_histologic_grade"] = X_out.apply(impute_grade, axis=1)
        if "tumor_stage" in X_out.columns:
            X_out["tumor_stage"] = X_out.apply(impute_stage, axis=1)
        return X_out

class DiscordantSignatureAdder(BaseEstimator, TransformerMixin):
    """Adds a custom discordant molecular score based on specific target genes.

    Calculates a score using 'mmp11', 'col22a1', and 'stat5a'. Validates the
    presence of these genes strictly to avoid silent failures.

    Raises:
        ValueError: If required target genes are missing from the dataframe.
    """
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        required_genes = ['stat5a', 'mmp11', 'col22a1']
        
        missing_genes = [gene for gene in required_genes if gene not in X_out.columns]
        if missing_genes:
            raise ValueError(f"Missing genes for Discordant Score: {missing_genes}")
            
        X_out['discordant_molecular_score'] = (
            X_out['mmp11'] + X_out['col22a1'] - X_out['stat5a']
        )
        return X_out

def build_preprocessor() -> Pipeline:
    """Returns a simple, clean pipeline."""
    return Pipeline(steps=[
        ("signature", DiscordantSignatureAdder()),
        ("smart_imputer", SmartClinicalImputer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
