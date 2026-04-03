import pandas as pd
import kagglehub
import os

def load_metabric_data():
    print("Fetching METABRIC dataset from Kaggle...")
    dataset_dir = kagglehub.dataset_download("raghadalharbi/breast-cancer-gene-expression-profiles-metabric")
    
    csv_path = os.path.join(dataset_dir, "METABRIC_RNA_Mutation.csv")
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    return df

def count_feature_categories(feature_columns):
    clinical_cols = []
    mutation_cols = []
    mrna_cols = []
    
    known_clinical = {
        'age_at_diagnosis', 'chemotherapy', 'cohort', 
        'neoplasm_histologic_grade', 'hormone_therapy', 
        'lymph_nodes_examined_positive', 'mutation_count', 
        'nottingham_prognostic_index', 'radio_therapy', 
        'tumor_size', 'tumor_stage'
    }
    
    for col in feature_columns:
        if col.endswith('_mut'):
            mutation_cols.append(col)
        elif col in known_clinical:
            clinical_cols.append(col)
        else:
            # If it is neither a mutation nor clinical, it is a gene's mRNA z-score
            mrna_cols.append(col)
            
    counts = {
        'Clinical Attributes': len(clinical_cols),
        'm-RNA levels z-score': len(mrna_cols),
        'Mutation': len(mutation_cols)
    }
    
    for category, count in counts.items():
        print(f"{category}: {count}")
        
    return counts, clinical_cols, mrna_cols, mutation_cols
