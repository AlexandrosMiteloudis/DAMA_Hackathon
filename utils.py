import pandas as pd
import kagglehub
import os

def load_metabric_data():
    print("Fetching METABRIC dataset from Kaggle...")
    dataset_dir = kagglehub.dataset_download("raghadalharbi/breast-cancer-gene-expression-profiles-metabric")
    
    csv_path = os.path.join(dataset_dir, "METABRIC_RNA_Mutation.csv")
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    return df
