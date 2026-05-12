import pandas as pd
import os

def ingest_raw_csvs(csv_paths_dict, intermediate_dir):
    """
    Reads the raw CSV files, converts them to Parquet for faster I/O, 
    saves them in the intermediate directory, and returns their path.
    """
    os.makedirs(intermediate_dir, exist_ok=True)
    parquet_paths = {}
    
    for name, path in csv_paths_dict.items():
        df = pd.read_csv(path)
        out_path = os.path.join(intermediate_dir, f"{name}.parquet")
        df.to_parquet(out_path)
        parquet_paths[name] = out_path
        
    return parquet_paths