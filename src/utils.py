from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple

import kagglehub
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from .competitors import get_rezaeian_features, get_kurniadi_features
from .mappings import KNOWN_CLINICAL_FEATURES


def load_metabric_data() -> pd.DataFrame:
    """Downloads and loads the METABRIC dataset from Kaggle applying minimal preprocessing.

    Fetches the breast cancer gene expression dataset containing RNA and
    mutation data for ~2,000 primary breast cancer samples.

    Returns:
        - A pandas DataFrame with METABRIC clinical, mutation, and RNA data.
        - A pandas Dataframe with some initial preprocrssing applied for the purposes of our paper.
    """
    print('Fetching METABRIC dataset from Kaggle...\n')

    dataset_dir = kagglehub.dataset_download(
        'raghadalharbi/breast-cancer-gene-expression-profiles-metabric'
    )

    csv_path = Path(dataset_dir) / 'METABRIC_RNA_Mutation.csv'

    df_original = pd.read_csv(csv_path, low_memory=False)
    print(f"Original dataset was loaded, shape: {df_original.shape}")
          
    # Minimal preprocessing
    df = df_original.copy()


    # Subtype filtering
    df = df_original.copy()
    df = df[df['pam50_+_claudin-low_subtype'] == 'LumA'].copy()

    # Target encoding
    def _encode_mortality(value) -> int:
        if str(value).strip().lower() == "died of disease":
            return 1
        return 0

    df['target_mortality'] = df['death_from_cancer'].apply(_encode_mortality)

    # Leakage removal
    leakage_cols = [
        'patient_id', 'overall_survival_months', 'overall_survival',
        'death_from_cancer', 'chemotherapy',
        'hormone_therapy', 'radio_therapy', 'type_of_breast_surgery'
    ]
    df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

    # Redundant columns
    remove_cols = [
        "cancer_type", "pam50_+_claudin-low_subtype"
    ]
    df.drop(columns=[c for c in remove_cols if c in df.columns], inplace=True)

    print(f"Original dataset was preprocessed, given a new shape: {df.shape}")

    return df_original, df



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

def build_metabric_pipeline(df: pd.DataFrame, kind: str = "") -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares features and target variable for the METABRIC dataset.

    Given subset of data related to the Luminal A subtype instances,
    it separates target variables to prevent data leakage, and either keeps numerical columns (kind == "pure_numerical")
    or encodes specific categorical columns, applies one-hot encoding to any remaining text variables.

    Args:
        df: The raw METABRIC pandas DataFrame.
        kind (optional): The type of pipeline to build ("pure_numerical" or other).

    Returns:
        A tuple containing:
            - X: The processed feature matrix (pandas DataFrame) ready for ML.
            - y: The target mortality variable (pandas Series).
    """
    df = df.copy()

    X = df.drop(columns=['target_mortality'])
    y = df['target_mortality']

    original_column_names = X.columns.tolist()

    if "cellularity" in X.columns:
        X["cellularity"] = X["cellularity"].map({"Low": 0, "Moderate": 1, "High": 2})


    if kind == "pure_numerical":

        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X = X[numeric_cols]
        categorical_cols = []

    else:

        # Encoding specific known categorical features
        for col in ["er_status", "her2_status", "pr_status"]:
            if col in X.columns:
                X[col] = (X[col] == "Positive").astype(int)

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
            print("\nApplying one-hot encoding to categorical columns: ")
            print(pd.Series({col: f"{X[col].nunique()} levels" for col in categorical_cols}))        
            print("\n---------\n")
            
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
            print("\nNew columns after one-hot encoding:", *X.columns.difference(original_column_names).tolist(), sep="\n")

        non_numeric = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric) > 0:
            raise ValueError(f"Pipeline error: Non-numeric columns remain and would be silently dropped: {list(non_numeric)}")

    # out of if-else block
    return X, y, original_column_names, categorical_cols


class SmartClinicalImputer(BaseEstimator, TransformerMixin):
    """Imputes missing clinical data (Grade and Stage) using medical heuristics.

    Uses Nottingham Prognostic Index (NPI) for grade imputation and tumor size +
    lymph nodes for stage imputation. Falls back to population medians if
    heuristics cannot be applied.

    Raises:
        ValueError: If required columns are missing during fitting.
    """

    def __init__(self):
        self._transform_output = "default"
        self.median_grade = None
        self.median_stage = None

    def set_output(self, *, transform=None):
        """Configure output container for sklearn compatibility.

        Args:
            transform: Output format, e.g. "default" or "pandas".

        Returns:
            Self.
        """
        self._transform_output = transform
        return self

    def fit(self, X, y=None):
        """Fit imputation medians on training data.

        Args:
            X: Input feature matrix.
            y: Optional target vector.

        Returns:
            Self.
        """
        if isinstance(X, pd.DataFrame):
            if "neoplasm_histologic_grade" not in X.columns:
                raise ValueError("Missing neoplasm_histologic_grade column")
            self.median_grade = X["neoplasm_histologic_grade"].median()

            if "tumor_stage" not in X.columns:
                raise ValueError("Missing tumor_stage column")
            self.median_stage = X["tumor_stage"].median()

        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        """Apply clinical imputation heuristics.

        Args:
            X: Input feature matrix.

        Returns:
            Imputed feature matrix.
        """
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
            if pd.isna(size) or pd.isna(ln):
                return self.median_stage
            if size > 50.0 or ln >= 4:
                return 3.0
            if size <= 20.0 and ln == 0:
                return 1.0
            return 2.0

        if "neoplasm_histologic_grade" in X_out.columns:
            X_out["neoplasm_histologic_grade"] = X_out.apply(impute_grade, axis=1)
        if "tumor_stage" in X_out.columns:
            X_out["tumor_stage"] = X_out.apply(impute_stage, axis=1)

        if self._transform_output == "pandas":
            return X_out

        return X_out

    def get_feature_names_out(self, input_features=None):
        """Return output feature names.

        Args:
            input_features: Optional input feature names.

        Returns:
            Output feature names (unchanged schema).
        """
        input_features = input_features or getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise ValueError(
                "input_features is required before fitting the transformer."
            )
        return list(input_features)

class DiscordantSignatureAdder(BaseEstimator, TransformerMixin):
    """Add a discordant molecular score based on target genes.

    Calculates:
        mmp11 + col22a1 - stat5a

    Raises:
        ValueError: If required target genes are missing from the input data.
    """

    def __init__(self):
        self._transform_output = "default"

    def set_output(self, *, transform=None):
        """Set output container configuration for sklearn compatibility.

        Args:
            transform: Output format requested by sklearn, e.g. "default"
                or "pandas".

        Returns:
            Self.
        """
        self._transform_output = transform
        return self

    def fit(self, X, y=None):
        """Fit the transformer.

        Args:
            X: Input feature matrix.
            y: Optional target vector.

        Returns:
            Self.
        """
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        """Add the discordant molecular score to the input data.

        Args:
            X: Input feature matrix.

        Returns:
            Transformed feature matrix with the added score column.
        """
        X_out = X.copy()
        required_genes = ["stat5a", "mmp11", "col22a1"]

        missing_genes = [gene for gene in required_genes if gene not in X_out.columns]
        if missing_genes:
            raise ValueError(f"Missing genes for Discordant Score: {missing_genes}")

        X_out["discordant_molecular_score"] = (
            X_out["mmp11"] + X_out["col22a1"] - X_out["stat5a"]
        )

        if self._transform_output == "pandas":
            return X_out

        return X_out

    def get_feature_names_out(self, input_features=None):
        """Return output feature names.

        Args:
            input_features: Optional input feature names.

        Returns:
            Output feature names including the engineered feature.
        """
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)

        if input_features is None:
            raise ValueError("input_features is required before fitting the transformer.")

        return list(input_features) + ["discordant_molecular_score"]
    
def add_clean_columns_to_pipeline(pipeline: Pipeline) -> Pipeline:
    """Add a custom step to pipeline that converts 'feature__col' to 'col' names.

    Args:
        pipeline: Input pipeline with ColumnTransformer.

    Returns:
        Pipeline with added 'clean_columns' step.
    """
    col_transform = pipeline.named_steps['col_transform']
    correct_columns = col_transform.get_feature_names_out()
    clean_columns = [col.split('__')[-1] for col in correct_columns]

    class CleanColumnNames(BaseEstimator, TransformerMixin):
        """Custom transformer to assign cleaned column names."""
        def __init__(self, columns):
            self.columns = columns

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return pd.DataFrame(X, columns=self.columns)

    return Pipeline(
        pipeline.steps + [('clean_columns', CleanColumnNames(clean_columns))]
    )

def build_preprocessor_with_augmentation(X: pd.DataFrame) -> Pipeline:
    """Build a preprocessing pipeline with feature augmentation and clean column names.

    The pipeline:
    1. Adds the discordant molecular score.
    2. Applies smart clinical imputation.
    3. Imputes and scales continuous features.
    4. Imputes dummy/binary features without scaling.
    5. Returns pandas DataFrames with readable column names.

    Args:
        X: Input feature matrix used to infer continuous and dummy columns.

    Returns:
        A scikit-learn Pipeline configured to output pandas DataFrames.
    """
    continuous_cols = [col for col in X.columns if X[col].nunique() > 2]
    dummy_cols = [col for col in X.columns if col not in continuous_cols]

    continuous_cols = continuous_cols + ["discordant_molecular_score"]

    missing_values = X.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    print("Missing columns report:\n", missing_cols)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, continuous_cols),
            ("cat", categorical_transformer, dummy_cols),
        ],
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(steps=[
        ("signature", DiscordantSignatureAdder()),
        ("smart_imputer", SmartClinicalImputer()),
        ("col_transform", col_transformer),
    ])

    return pipeline.set_output(transform="pandas")


def build_preprocessor_pure_numerical(X: pd.DataFrame) -> Pipeline:
    """Returns a simple pipeline that scales all features without augmentation."""

    # we impute missing numeric values with the median of that specific feature.
    missing_values = X.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    print("Missing columns report: \n", missing_cols)


    # detect numerical columns
    numeric_columns = X.select_dtypes(include="number").columns

    # introduce preprocessing pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns)
        ],
        remainder="passthrough"
    )

    preprocessor.set_output(transform="pandas")
    preprocessor.verbose_feature_names_out = False
    

    return Pipeline(steps=[
        ("num_transform", preprocessor)
    ])



def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into stratified train and test sets and print class balance.

    Args:
        X: Feature matrix.
        y: Target vector used both as labels and for stratification.
        test_size: Proportion of samples assigned to the test split.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple containing:
            - X_train: Training feature matrix.
            - X_test: Test feature matrix.
            - y_train: Training target vector.
            - y_test: Test target vector.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train target distribution:\n{y_train.value_counts()}")
    print(f"Test  target distribution:\n{y_test.value_counts()}")

    return X_train, X_test, y_train, y_test



def prepare_data_splits_per_model(
    approach: str,
    df: pd.DataFrame,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[Any, Any, pd.Series, pd.Series, Any]:
    """Prepare train/test data and preprocessing pipeline for a modeling approach.

    This function selects features and target based on the chosen approach,
    performs a stratified train/test split, fits the preprocessor on the
    training data only, and transforms both train and test sets.

    Args:
        approach: Modeling approach name. Supported values are
            "pure_numerical", "with_augmentation", "rezaeian", and "kurniadi".
        df: Input METABRIC dataframe containing features and target.
        test_size: Proportion of samples assigned to the test set.
        random_state: Random seed for reproducible splitting.

    Returns:
        A tuple (X_train_processed, X_test_processed, y_train, y_test, preprocessor),
        where X_train_processed and X_test_processed are transformed feature
        matrices, y_train and y_test are target vectors, and preprocessor is
        the fitted preprocessing object.

    Raises:
        ValueError: If `approach` is not supported.
    """
    if approach == "pure_numerical":
        X, y, _, _ = build_metabric_pipeline(df, kind="pure_numerical")
        preprocessor_builder = build_preprocessor_pure_numerical

    elif approach == "with_augmentation":
        X, y, _, _ = build_metabric_pipeline(df)
        preprocessor_builder = build_preprocessor_with_augmentation
        #preprocessor_builder = add_clean_columns_to_pipeline(preprocessor_builder)

    elif approach == "rezaeian":
        feature_list = get_rezaeian_features()
        X = df[feature_list]
        y = df["target_mortality"]
        preprocessor_builder = build_preprocessor_pure_numerical

    elif approach == "kurniadi":
        feature_list = get_kurniadi_features()
        X = df[feature_list]
        y = df["target_mortality"]
        preprocessor_builder = build_preprocessor_pure_numerical

    else:
        raise ValueError(
            f"Unknown approach: {approach}. "
            "Supported approaches are: "
            "'pure_numerical', 'with_augmentation', 'rezaeian', 'kurniadi'."
        )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    #preprocessor = preprocessor_builder(X_train)

    #X_train_processed = preprocessor.fit_transform(X_train)
    #X_test_processed = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor_builder