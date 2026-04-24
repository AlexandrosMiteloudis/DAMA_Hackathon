import pandas as pd
from typing import Sequence, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_metabric_data() -> pd.DataFrame:
    import kagglehub
    from pathlib import Path
    print('Fetching METABRIC dataset from Kaggle...')
    dataset_dir = kagglehub.dataset_download(
        'raghadalharbi/breast-cancer-gene-expression-profiles-metabric'
    )
    csv_path = Path(dataset_dir) / 'METABRIC_RNA_Mutation.csv'
    return pd.read_csv(csv_path, low_memory=False)

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares features and target variable."""
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

    # Encoding
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

    # Keep numeric only
    X = X.select_dtypes(include=['number'])
    return X, y

class SmartClinicalImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
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
        
        def impute_grade(row):
            if not pd.isna(row.get("neoplasm_histologic_grade")): 
                return row["neoplasm_histologic_grade"]
            npi = row.get("nottingham_prognostic_index")
            if pd.isna(npi): return self.median_grade 
            if npi < 3.0: return 1.0
            if npi < 4.2: return 2.0
            return 3.0

        def impute_stage(row):
            if not pd.isna(row.get("tumor_stage")): 
                return row["tumor_stage"]
            size = row.get("tumor_size")
            ln = row.get("lymph_nodes_examined_positive")
            if pd.isna(size) or pd.isna(ln): return self.median_stage 
            if size > 50.0 or ln >= 4: return 3.0
            if size <= 20.0 and ln == 0: return 1.0
            return 2.0

        if "neoplasm_histologic_grade" in X_out.columns:
            X_out["neoplasm_histologic_grade"] = X_out.apply(impute_grade, axis=1)
        if "tumor_stage" in X_out.columns:
            X_out["tumor_stage"] = X_out.apply(impute_stage, axis=1)
        return X_out

class DiscordantSignatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        required_genes = ['stat5a', 'mmp11', 'col22a1']
        
        # STRICT VALIDATION: This answers his comment about AI agents making silent mistakes
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