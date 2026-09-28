import json
import os
from typing import List, Dict, Any

class ResultWriter:
    """
    Handles deterministic serialization of evaluation results.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def write(self, dataset_name: str, run_id: str, config: dict, session_results: List[Dict[str, Any]]) -> str:
        """
        Writes the results to a JSON file.
        """
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        file_path = os.path.join(dataset_dir, f"{run_id}_evaluation.json")
        
        # Create a deterministic structure
        payload = {
            "dataset": dataset_name,
            "run_id": run_id,
            "seed": config.get("seed", 42),
            "configuration": config,
            "provenance": {
                "accuracy": "PAPER-FACT: Cumulative accuracy over all seen classes.",
                "dataset_split": "IMPLEMENTATION-CHOICE: Pseudo-random deterministic permutation.",
                "preprocessing": "IMPLEMENTATION-CHOICE: Standard CLIP resize and normalization."
            },
            "results": session_results
        }
        
        with open(file_path, "w") as f:
            json.dump(payload, f, indent=4)
            
        return file_path
