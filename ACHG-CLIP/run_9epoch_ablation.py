import os
import subprocess
import yaml
import shutil
import time
from run_experiments import modify_yaml

def main():
    print("==============================================")
    print("Starting 9-Epoch Base Training Ablation")
    print("==============================================")

    cifar100_yaml = "configs/data/cifar100.yaml"
    clip_yaml = "configs/model/clip_backbone.yaml"

    # Backup original configs
    shutil.copy(cifar100_yaml, cifar100_yaml + ".bak")
    shutil.copy(clip_yaml, clip_yaml + ".bak")

    try:
        # 1. Setup exact configurations for Backbone_CLIP_B
        modify_yaml(cifar100_yaml, "seed", 42)
        modify_yaml(clip_yaml, "variant", "ViT-B/16")
        
        # 2. Run script with resume argument from the 6-epoch run
        checkpoint_path = r"results\run_31082026_195308\latest_checkpoint.pt"
        if not os.path.exists(checkpoint_path):
            print(f"Error: Could not find base checkpoint at {checkpoint_path}")
            return
            
        print(f"Resuming from: {checkpoint_path}")
        
        subprocess.run(["python", "run_cifar100.py", "--resume_base", checkpoint_path], check=True)
        
    finally:
        # Restore baseline configs
        print("Restored baseline configs.")
        shutil.move(cifar100_yaml + ".bak", cifar100_yaml)
        shutil.move(clip_yaml + ".bak", clip_yaml)

if __name__ == "__main__":
    main()
