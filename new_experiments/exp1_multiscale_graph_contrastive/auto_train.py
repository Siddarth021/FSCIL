import subprocess
import sys
import re
import time

def main():
    print("Starting autonomous training and monitoring for CIFAR-100...")
    
    # Baseline expectations
    BASE_MIN_ACC = 0.85  # We expect at least 85% on Session 0
    
    process = subprocess.Popen(
        [sys.executable, "run_cifar100.py", "--seed", "202"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    session_0_acc = None
    
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            
            # Check for Session 0 accuracy
            if "Epoch" in line and "Accuracy:" in line:
                # e.g., "Epoch 3 Accuracy: 0.8620"
                match = re.search(r"Accuracy:\s+([0-9\.]+)", line)
                if match:
                    acc = float(match.group(1))
                    if session_0_acc is None or acc > session_0_acc:
                        session_0_acc = acc
            
            # Check if Session 0 finished
            if "Session 0 Training completed" in line:
                if session_0_acc is not None and session_0_acc < BASE_MIN_ACC:
                    print(f"\n[AUTO-STOP] Session 0 accuracy ({session_0_acc:.4f}) is below baseline minimum ({BASE_MIN_ACC:.4f}). Aborting run to save time.")
                    process.terminate()
                    sys.exit(1)
                else:
                    print(f"\n[AUTO-CONTINUE] Session 0 accuracy ({session_0_acc:.4f}) meets baseline minimum. Continuing to incremental sessions.")
                    
            # Check for cumulative accuracy drops
            if "Cumulative accuracy:" in line:
                match = re.search(r"Cumulative accuracy:\s+([0-9\.]+)", line)
                if match:
                    acc = float(match.group(1))
                    # If it drops below 70% at any point, it's definitely worse than baseline
                    if acc < 0.70:
                        print(f"\n[AUTO-STOP] Incremental accuracy dropped too low ({acc:.4f}). Aborting run to save time.")
                        process.terminate()
                        sys.exit(1)
                        
    except KeyboardInterrupt:
        process.terminate()
        sys.exit(1)
        
    process.wait()
    if process.returncode != 0:
        print(f"\n[AUTO-STOP] Training script exited with error code {process.returncode}.")
        sys.exit(process.returncode)
        
    print("\n[AUTO-FINISH] Training completed.")

if __name__ == "__main__":
    main()
