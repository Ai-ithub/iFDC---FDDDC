import os
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
from pathlib import Path

def setup_directories():
    """Create required folders"""
    Path("datasets/processed").mkdir(exist_ok=True)
    Path("datasets/outliers").mkdir(exist_ok=True)
    Path("datasets/missing").mkdir(exist_ok=True)  # New folder for missing data
    Path("datasets/stats").mkdir(exist_ok=True)

def detect_outliers(df, method='iqr', threshold=3):
    """
    Detect outliers using different methods
    Parameters:
        method: 'iqr' or 'zscore'
        threshold: threshold for outlier detection
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'iqr':
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[numeric_cols] < (Q1 - threshold*IQR)) | 
                    (df[numeric_cols] > (Q3 + threshold*IQR))).any(axis=1)
    
    elif method == 'zscore':
        z_scores = stats.zscore(df[numeric_cols], nan_policy='omit')
        outliers = (np.abs(z_scores) > threshold).any(axis=1)
    
    return outliers

def standardize_data(df, method='standard'):
    """
    Standardize data
    Parameters:
        method: 'standard' (mean-std) or 'minmax'
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        stats = {
            'method': 'standard',
            'mean': df[numeric_cols].mean().to_dict(),
            'std': df[numeric_cols].std().to_dict()
        }
    elif method == 'minmax':
        min_vals = df[numeric_cols].min()
        max_vals = df[numeric_cols].max()
        df[numeric_cols] = (df[numeric_cols] - min_vals) / (max_vals - min_vals)
        stats = {
            'method': 'minmax',
            'min': min_vals.to_dict(),
            'max': max_vals.to_dict()
        }
    
    return df, stats

def process_partition(df, outlier_method='iqr', standardization_method='standard'):
    """Process each data partition"""
    # 1. Detect and separate missing data
    missing_mask = df.isnull().any(axis=1)
    df_missing = df[missing_mask]
    df = df[~missing_mask]
    
    # 2. Detect outliers
    outliers = detect_outliers(df, method=outlier_method)
    
    # 3. Standardize data
    df_clean, scaling_stats = standardize_data(df[~outliers], method=standardization_method)
    
    return {
        'clean': df_clean,
        'outliers': df[outliers],
        'missing': df_missing,
        'stats': scaling_stats
    }

def main():
    setup_directories()
    
    with Client(n_workers=4, memory_limit='4GB') as client:
        # Read data
        ddf = dd.read_parquet("datasets/synthetic_fdms_chunks/*.parquet", chunksize=100000)
        
        # Parallel processing
        results = ddf.map_partitions(
            process_partition,
            outlier_method='zscore',  # or 'iqr'
            standardization_method='standard'  # or 'minmax'
        ).compute()
        
        # Aggregate results
        df_clean = dd.concat([r['clean'] for r in results])
        df_outliers = dd.concat([r['outliers'] for r in results])
        df_missing = dd.concat([r['missing'] for r in results])  # missing data
        
        # Save results
        df_clean.to_parquet("datasets/processed/", partition_on=['WELL_ID'])
        df_outliers.to_parquet("datasets/outliers/")
        df_missing.to_parquet("datasets/missing/")  # save missing data
        
        # Save scaling statistics
        with open("datasets/stats/scaling_stats.json", "w") as f:
            json.dump(results[0]['stats'], f)  # using first partition's stats as a sample

if __name__ == '__main__':
    main()
