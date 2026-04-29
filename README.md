# Exploring METABRIC dataset

Binary classification pipeline for predicting breast cancer mortality in Luminal A patients using the [METABRIC](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric) dataset. The project compares standard ML models, interpretable models, and published gene-signature baselines across multiple feature spaces.

## Preprocessing

Data loading (`load_metabric_data`) applies the following steps before any modeling:

1. **Subtype filtering** — Only Luminal A (`LumA`) patients are kept (~680 samples).
2. **Target encoding** — `death_from_cancer == "Died of Disease"` → 1, everything else → 0 (stored as `target_mortality`). This produces an imbalanced binary target (~21.5% positive class).
3. **Leakage removal** — Columns that would leak outcome information are dropped: `patient_id`, `overall_survival_months`, `overall_survival`, `death_from_cancer`, `chemotherapy`, `hormone_therapy`, `radio_therapy`, `type_of_breast_surgery`.
4. **Redundant column removal** — `cancer_type` and `pam50_+_claudin-low_subtype` (constant after filtering) are dropped.

The feature matrix then goes through one of two preprocessing pipelines depending on the approach:

| Pipeline | Steps | Used by |
|---|---|---|
| **`with_augmentation`** | Discordant molecular score (`mmp11 + col22a1 − stat5a`), smart clinical imputation (NPI-based grade, tumor-size-based stage), categorical encoding (ER/HER2/PR → binary, mutations → binary, remaining categoricals → one-hot), median imputation + standard scaling for continuous features | `with_augmentation` approach |
| **`pure_numerical`** | Keep only pre-existing numeric columns, median imputation + standard scaling | `pure_numerical`, `rezaeian`, `kurniadi` approaches |

Custom transformers in `src/utils.py`:
- **`SmartClinicalImputer`** — Uses Nottingham Prognostic Index to impute missing histologic grade, and tumor size + lymph node count to impute missing tumor stage. Falls back to population median when heuristics cannot be applied.
- **`DiscordantSignatureAdder`** — Adds an engineered feature `discordant_molecular_score = mmp11 + col22a1 − stat5a`, capturing a discordant molecular signal associated with unexpected mortality in low-risk patients.

## Competitor Baselines

Published gene signatures are used as baselines to benchmark our feature spaces. The logic is in `src/competitors.py`.

| Baseline | Reference | Features | Description |
|---|---|---|---|
| **Rezaeian et al.** | [doi:10.12688/f1000research.9417.1](https://doi.org/10.12688/f1000research.9417.1) | 39 genes | mRNA z-score features only — no clinical attributes, no mutations. Genes are mapped to METABRIC columns by name, with mRNA prioritized over mutation suffix. |
| **Kurniadi & Saputri** | [doi:10.1109/ICIMTECH63123.2024.10780791](https://doi.org/10.1109/ICIMTECH63123.2024.10780791) | 15 genes | Compact mRNA-only signature. Same column-mapping logic as Rezaeian. |

`map_competitor_features()` resolves each gene name against available columns: it first looks for the mRNA column, then falls back to the `_mut` (mutation) column if the mRNA is absent.

## Feature-Space Analysis

Each modeling approach operates on a different subset of the METABRIC feature space. The `feature_space_analysis.ipynb` notebook categorizes columns into three groups — **Clinical Attributes**, **mRNA z-scores**, and **Mutation flags** — using `count_feature_categories()`.

| Approach | Total features | Clinical | mRNA | Mutation | Notes |
|---|---|---|---|---|---|
| **`with_augmentation`** | 683 | 21 | 489 | 173 | Full feature space with one-hot encoded categoricals + engineered discordant score, leading to 706 after the encoding and augmentation |
| **`pure_numerical`** | 498 | 9 | 489 | 0 | Only pre-existing numeric columns; no categorical encoding, no mutation flags |
| **`rezaeian`** | 39 | 0 | 39 | 0 | Published 39-gene mRNA-only signature |
| **`kurniadi`** | 15 | 0 | 15 | 0 | Published 15-gene mRNA-only signature |

Key observations:
- The competitor baselines use **exclusively mRNA z-score** features — no clinical context, no mutation data.
- The `pure_numerical` approach retains clinical numerics (age, tumor size, NPI, grade, stage, lymph nodes) alongside mRNA z-scores, but drops non-numeric columns (e.g., ER/HER2/PR status as text) and mutation flags (stored as text).
- The `with_augmentation` approach is the richest, encoding categoricals and mutations as binary features and adding the engineered discordant molecular score.
- Despite the large dimensionality gap, compact signatures (15–39 genes) can be competitive due to reduced noise and overfitting risk.

## Repository Structure

```
├── data/
│   ├── METABRIC_RNA_Mutation.csv            # Raw METABRIC dataset (clinical + mRNA + mutation)
│   └── best_config_per_experiment.csv       # Exported best model configs per experiment
├── notebooks/
│   ├── final_experiments.ipynb              # Main experiment notebook — runs all approaches, threshold tuning, and leaderboard export
│   ├── METABRIC_CIBCB_pipeline.ipynb        # End-to-end pipeline used for the CIBCB paper
│   ├── METABRIC_interpretable_models.ipynb  # Benchmarks interpretable models (EBM, FIGS, RuleFit, etc.)
│   ├── METABRIC_Preprocessing_Pipelines.ipynb # Exploration of preprocessing strategies
│   ├── feature_space_analysis.ipynb         # Analysis of different feature spaces and their properties
│   ├── unified_pipeline.ipynb               # Earlier unified pipeline prototype
│   └── test_utils.ipynb                     # Quick tests for src utility functions
├── src/
│   ├── models.py        # Model registries with hyperparameter search spaces (sklearn + interpretable)
│   ├── utils.py         # Data loading, preprocessing pipelines, train/test splitting, custom transformers
│   ├── competitors.py   # Competitor gene signatures (Rezaeian, Kurniadi) and evaluation helpers
│   └── mappings.py      # Constants and feature-name mappings (clinical feature sets)
├── requirements.txt
└── LICENSE              # MIT
```

## Installing requirements

To ensure your environment matches the project requirements, run:

```bash
conda create -n <your-preferred-env-name> python==3.12.0
conda activate <your-preferred-env-name>
pip install -r requirements.txt
```

The above mentioned setup has been tested on macOS (Tahoe 26.4.1) and Ubuntu (24.04.4 LTS).
 
