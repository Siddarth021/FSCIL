import json
import os

with open("f:/FSCIL/ACHG-CLIP/run_cifar100.py", "r", encoding="utf-8") as f:
    run_cifar100_content = f.read()

with open("f:/FSCIL/ACHG-CLIP/run_experiments.py", "r", encoding="utf-8") as f:
    run_experiments_content = f.read()

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ACHG-CLIP: Extended Base (6-Epoch)\n",
            "Resumes from the 3rd epoch checkpoint of the ViT-L/14 B2 study and trains for 3 more epochs (6 total), followed by incremental sessions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118\n",
            "!pip install -q transformers ftfy regex pyyaml \"numpy<2\"\n",
            "\n",
            "import os\n",
            "import glob\n",
            "if not os.path.exists('/kaggle/working/FSCIL'):\n",
            "    !git clone https://github.com/Siddarth021/FSCIL.git /kaggle/working/FSCIL\n",
            "\n",
            "os.chdir('/kaggle/working/FSCIL/ACHG-CLIP')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Write latest run_cifar100.py\n",
            f"code_cifar = {json.dumps(run_cifar100_content)}\n",
            "with open('run_cifar100.py', 'w', encoding='utf-8') as f:\n",
            "    f.write(code_cifar)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess\n",
            "import sys\n",
            "import glob\n",
            "\n",
            "# Find the checkpoint from the previous kernel\n",
            "checkpoints = glob.glob('/kaggle/input/**/latest_checkpoint.pt', recursive=True)\n",
            "if not checkpoints:\n",
            "    raise ValueError('Could not find latest_checkpoint.pt in /kaggle/input/')\n",
            "resume_path = checkpoints[0]\n",
            "print(f'Resuming from {resume_path}')\n",
            "\n",
            "print('Starting Extended Base 6-Epoch Training...')\n",
            "cmd = [sys.executable, 'run_cifar100.py', '--variant', 'ViT-L/14', '--seed', '42', '--data_root', './datasets', '--resume_base', resume_path, '--base_epochs', '3']\n",
            "\n",
            "process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
            "for line in process.stdout:\n",
            "    print(line, end='', flush=True)\n",
            "process.wait()\n",
            "\n",
            "if process.returncode != 0:\n",
            "    raise RuntimeError(f'Training failed with exit code {process.returncode}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import json\n",
            "import shutil\n",
            "\n",
            "results_dir = 'results'\n",
            "subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('run_')]\n",
            "if subdirs:\n",
            "    latest_run = max(subdirs, key=os.path.getmtime)\n",
            "    print(f'Latest run folder: {latest_run}')\n",
            "    summary_file = os.path.join(latest_run, 'eval_summary.json')\n",
            "    if os.path.exists(summary_file):\n",
            "        with open(summary_file, 'r') as f:\n",
            "            summary = json.load(f)\n",
            "        print('\\n' + '='*60)\n",
            "        print('FINAL EVALUATION SUMMARY')\n",
            "        print('='*60)\n",
            "        print(json.dumps(summary, indent=2))\n",
            "        shutil.copy(summary_file, '/kaggle/working/eval_summary.json')\n",
            "    \n",
            "    dest = '/kaggle/working/Backbone_CLIP_C_ViTL14_Extended6'\n",
            "    if os.path.exists(dest):\n",
            "        shutil.rmtree(dest)\n",
            "    shutil.copytree(latest_run, dest)\n",
            "    print(f'\\nArtifacts saved to {dest}')"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

os.makedirs("f:/FSCIL/kaggle_runner_6epoch", exist_ok=True)
with open("f:/FSCIL/kaggle_runner_6epoch/achg_clip_vit_large_6epoch.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created f:/FSCIL/kaggle_runner_6epoch/achg_clip_vit_large_6epoch.ipynb successfully!")
