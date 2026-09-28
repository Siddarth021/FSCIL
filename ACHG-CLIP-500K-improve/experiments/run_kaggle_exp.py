"""
Kaggle Kernel Entry Point for Experiments 2, 3, 4, 5
=====================================================
This file acts as the main notebook cell code in Kaggle.
Set EXP_ID to 2, 3, 4, or 5 to run the corresponding experiment.
"""
import subprocess, sys, os, time

# ─── CONFIGURE WHICH EXP TO RUN ──────────────────────────────────────────────
EXP_ID = 2   # <-- CHANGE THIS TO 2, 3, 4, or 5
DATA_ROOT = "/kaggle/input/cifar100/data" if os.path.exists("/kaggle/input/cifar100") else "./datasets"
# ─────────────────────────────────────────────────────────────────────────────

EXP_SCRIPTS = {
    2: "experiments/exp2_label_smoothing/run_exp2.py",
    3: "experiments/exp3_strong_augment/run_exp3.py",
    4: "experiments/exp4_bidirectional_loss/run_exp4.py",
    5: "experiments/exp5_multi_template/run_exp5.py",
}

if EXP_ID not in EXP_SCRIPTS:
    raise ValueError(f"EXP_ID must be 2-5, got {EXP_ID}")

script = EXP_SCRIPTS[EXP_ID]
print(f"Starting EXP{EXP_ID}: {script}")
print(f"Data root: {DATA_ROOT}")

cmd = [
    sys.executable, script,
    "--data_root", DATA_ROOT,
    "--variant", "ViT-L/14"
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1)
try:
    for line in process.stdout:
        print(line, end="", flush=True)
except KeyboardInterrupt:
    process.terminate()
    sys.exit(1)

process.wait()
if process.returncode != 0:
    print(f"\n[FAILED] Script exited with code {process.returncode}")
    sys.exit(process.returncode)
print(f"\n[DONE] EXP{EXP_ID} completed successfully.")
