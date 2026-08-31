import yaml
import os
import subprocess
import shutil

def modify_yaml(filepath, key, value):
    with open(filepath, "r") as f:
        cfg = yaml.safe_load(f)
    if key in cfg:
        if isinstance(cfg[key], dict) and "value" in cfg[key]:
            cfg[key]["value"] = value
        else:
            cfg[key] = value
    else:
        cfg[key] = {"value": value}
    with open(filepath, "w") as f:
        yaml.dump(cfg, f)

def run_experiment(name, data_seed, backbone_variant):
    print(f"\n==============================================")
    print(f"Starting Experiment: {name}")
    print(f"==============================================")
    
    # 1. Modify configs
    modify_yaml("configs/data/cifar100.yaml", "seed", data_seed)
    modify_yaml("configs/model/clip_backbone.yaml", "variant", backbone_variant)
    
    # 2. Run the training script
    subprocess.run(["python", "run_cifar100.py"], check=True)
    
    # 3. Find the most recently created run directory in results/
    results_dir = "results"
    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("run_")]
    if subdirs:
        latest_run = max(subdirs, key=os.path.getmtime)
        
        # 4. Rename it to the experiment name for easy tracking
        new_dir = os.path.join(results_dir, name)
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        os.rename(latest_run, new_dir)
        print(f"Experiment {name} saved to {new_dir}\n")

if __name__ == "__main__":
    # Baseline configs to restore later
    original_seed = 42
    original_variant = "ViT-B/32"
    
    try:
        # Experiment A: Split B (Seed 1993, ViT-B/32)
        run_experiment("SplitStudy", 1993, "ViT-B/32")
        
        # Experiment B1: Backbone CLIP-B (Seed 42, ViT-B/16)
        run_experiment("Backbone_CLIP_B", 42, "ViT-B/16")
        
        # Experiment B2: Backbone CLIP-C (Seed 42, ViT-L/14)
        run_experiment("Backbone_CLIP_C", 42, "ViT-L/14")
        
    finally:
        # Always restore baseline configs
        modify_yaml("configs/data/cifar100.yaml", "seed", original_seed)
        modify_yaml("configs/model/clip_backbone.yaml", "variant", original_variant)
        print("Restored baseline configs.")
