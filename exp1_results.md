# Experiment 1 Results (Multi-scale Graph Contrastive)

The following table aggregates the session-wise accuracies across the evaluated random seeds for Experiment 1 (Multi-scale Graph Contrastive Learning).

| Seed | Session 0 | Session 1 | Session 2 | Session 3 | Session 4 | Session 5 | Session 6 | Session 7 | Session 8 | Mean Accuracy | Final Drop | lambda |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **123** (Kaggle) | 87.40% | 87.31% | 86.41% | 85.53% | 85.32% | 84.92% | 84.33% | 82.87% | 82.55% | 85.18% | -4.85% | 0.005 |
| **123** (Exp1A Temp=0.1) | 87.63% | 87.29% | 86.56% | 85.79% | 85.29% | 84.60% | 83.97% | 82.66% | 82.37% | 85.13% | -5.26% | 0.01 |
| **123** (Exp1B Heads) | 87.87% | 87.88% | 87.54% | 86.65% | 85.54% | 85.04% | 84.32% | 83.03% | 82.13% | 85.56% | -5.74% | 0.01 |
| **123** (Baseline Unmodified) | 87.42% | 86.74% | 86.17% | 85.31% | 85.20% | 84.40% | 83.83% | 83.06% | 82.81% | 84.99% | -4.61% | N/A |
| **123** (Exp1C Aggressive) | 87.20% | 87.05% | 86.17% | 85.08% | 84.38% | 84.05% | 83.29% | 82.48% | 82.35% | 84.67% | -4.85% | 0.05 |
| **123** (Exp1D Cross-Modal) | **88.07%** | **87.78%** | **87.69%** | **86.61%** | **86.10%** | **84.49%** | **84.21%** | **83.27%** | **82.69%** | **85.66%** | -5.38% | 0.03 |
| **123** (Exp1E True Multi-Scale) | 87.88% | 87.69% | 86.41% | 85.69% | 85.06% | 84.48% | 83.58% | 82.36% | 81.97% | 85.01% | -5.91% | 0.03 |
| **123** (Exp1F MultiScale + LS) | *Running* | - | - | - | - | - | - | - | - | - | - | 0.03 |
