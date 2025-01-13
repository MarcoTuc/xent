import os
import json
from tqdm import tqdm
from xent.config import *

def merge_datasets(base_dir="instruct_llama"):
    # Setup paths
    data_path = os.path.join(data_dir, base_dir)
    closure_dir = os.path.join(data_path, "closure")
    ranking_dir = os.path.join(data_path, "ranking")
    
    # Process closure files
    closure_database = []
    closure_files = [f for f in os.listdir(closure_dir) if f.endswith('.json')]
    print(f"Found {len(closure_files)} closure files")
    
    for file in tqdm(closure_files, desc="Processing closure files"):
        file_path = os.path.join(closure_dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            closure_database.extend(data)
    
    # Save closure database
    closure_output = os.path.join(data_path, "closure_database.json")
    print(f"Saving closure database with {len(closure_database)} entries to {closure_output}")
    with open(closure_output, 'w') as f:
        json.dump(closure_database, f)
    
    # Process ranking files
    ranking_database = []
    ranking_files = [f for f in os.listdir(ranking_dir) if f.endswith('.json')]
    print(f"Found {len(ranking_files)} ranking files")
    
    for file in tqdm(ranking_files, desc="Processing ranking files"):
        file_path = os.path.join(ranking_dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            ranking_database.extend(data)
    
    # Save ranking database
    ranking_output = os.path.join(data_path, "ranking_database.json")
    print(f"Saving ranking database with {len(ranking_database)} entries to {ranking_output}")
    with open(ranking_output, 'w') as f:
        json.dump(ranking_database, f)

if __name__ == "__main__":
    merge_datasets()