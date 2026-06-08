import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

def compute_mutation_volcano(df, mutations, target, min_mutation_count=5):
    """
    Computes association between binary mutations and a binary target 
    using Fisher's Exact Test.
    """
    results = []
    for col in mutations:
        mutation_count = df[col].sum()
        if mutation_count < min_mutation_count:
            continue

        ct = pd.crosstab(df[col], df[target])
        ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        
        a, b = ct.loc[0, 0], ct.loc[0, 1]
        c, d = ct.loc[1, 0], ct.loc[1, 1]

        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        _, p_value = fisher_exact([[a, b], [c, d]])

        results.append({
            "Mutation": col,
            "OddsRatio": odds_ratio,
            "P_Value": p_value,
            "Mutation_Count": mutation_count
        })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return res_df

    res_df["FDR"] = multipletests(res_df["P_Value"], method="fdr_bh")[1]
    res_df["Log2_OR"] = np.log2(res_df["OddsRatio"])
    res_df["Neg_Log10_FDR"] = -np.log10(res_df["FDR"])
    return res_df