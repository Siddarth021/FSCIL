import sys
import os
import yaml

# Add current path to sys.path
sys.path.append(os.getcwd())

from data.registry import get_data_manager

def check_dataset(config_path):
    dataset_name = os.path.basename(config_path).split('.')[0]
    print(f"\n--- Checking {dataset_name.upper()} ---")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Requesting synthetic=False to load real data
        manager = get_data_manager(config, data_root='D:/FSCIL/datasets', synthetic=False)
        print(f"SUCCESS: Dataset {dataset_name} loaded successfully.")
    except Exception as e:
        print(f"FAILED: Could not load {dataset_name}.")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")

if __name__ == "__main__":
    configs = [
        "configs/data/cifar100.yaml",
        "configs/data/mini_imagenet.yaml",
        "configs/data/cub200.yaml"
    ]
    for c in configs:
        check_dataset(c)
