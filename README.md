# Predicting Breast Cancer Survival: Clinical vs. Genomic Machine Learning
**An Integrated Machine Learning Approach for 5-Year Survival and Time-to-Event Analysis**

### Project Objective
This project investigates whether integrating a targeted 22-gene expression panel and genetic mutation data significantly improves breast cancer survival predictions compared to a strictly clinical baseline. 

We utilize the **METABRIC** (Molecular Taxonomy of Breast Cancer International Consortium) dataset to build predictive pipelines evaluating both binary 5-year survival (Classification) and exact time-to-event timelines (Survival Analysis).

### Methodology
1. **Clinical Baseline:** We established a predictive baseline using 11 standard clinical features (Age, Tumor Size, Histologic Grade, etc.).
2. **Genomic Integration:** We engineered a targeted biological panel consisting of 22 established breast cancer biomarker genes (e.g., *BRCA1, BRCA2, TP53, PIK3CA*) and significant mutations isolated via an annotated Volcano Plot.
3. **Machine Learning (Phase 1):** We conducted an Ablation Study using a Soft Voting Ensemble (Logistic Regression, Random Forest, Gradient Boosting) to compare Clinical vs. Genomic predictive power via ROC-AUC.
4. **Machine Learning (Phase 2):** We utilized `scikit-survival` (Cox Proportional Hazards, Random Survival Forests) to predict exact survival timelines, measured by the Concordance Index (C-Index).

### How to Run the Code
The complete dataset (`METABRIC_RNA_Mutation.csv`) is included directly in this repository for seamless replication. 

To run this project:
1. Clone this repository to your local machine or open the notebook in Google Colab.
2. Ensure the `METABRIC_RNA_Mutation.csv` file remains in the same directory as the `METABRICIA.ipynb` notebook.
3. Run all cells sequentially.

### Libraries Used
* `pandas`, `numpy`, `matplotlib`, `seaborn`
* `scikit-learn` (Pipelines, ColumnTransformers, Ensembles)
* `scikit-survival` (Time-to-Event Analysis)
* `lifelines` (Kaplan-Meier Modeling)
