# Apple Quality Prediction with Neural Networks and Optimization Algorithms

![MATLAB](https://img.shields.io/badge/MATLAB-R202x-orange)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-blue)
![Dataset](https://img.shields.io/badge/Dataset-CSV-lightgrey)

## Overview
This repository implements a controlled experimental pipeline for binary apple quality classification using a feed-forward neural network in MATLAB. The study compares gradient-based optimizers (GD, CG, BFGS, DFP) against a swarm-intelligence optimizer (ABC) under identical preprocessing, initialization, and evaluation conditions. The focus is on convergence behavior, generalization stability, and cross-run robustness.

## Problem Statement
Accurate quality grading of agricultural products influences post-harvest planning and market valuation. This project formulates apple quality prediction as a supervised binary classification task and evaluates how optimizer choice affects model accuracy, stability, and susceptibility to overfitting.

## Architecture Explanation
**High level explanation**  
The codebase is a monolithic MATLAB research pipeline organized into phase-based scripts. Each script handles data ingestion, preprocessing, training, evaluation, and visualization in a fully reproducible workflow. Experiments are deterministic per seed to enable fair cross-optimizer comparisons.

**Why this architecture was chosen**  
The goal is controlled experimentation rather than deployment. A single MATLAB pipeline reduces implementation variability, keeps numerical precision consistent, and allows direct comparison across optimizers without confounding differences in tooling.

## Tech Stack
- **Language:** MATLAB
- **ML/Math:** MATLAB built-in numerical and plotting libraries
- **Data Storage:** CSV file (`apple_quality.csv`)
- **Visualization:** MATLAB figures saved to local `figures/`

## Experimental Setup
- **Network architecture:** Single hidden-layer MLP. Hidden size varies by experiment (e.g., `H = 8, 16, 32`), output layer is 2 units for one-hot binary classification. Activation is `tanh` in hidden layer; output uses softmax.
- **Loss function:** Cross-entropy with optional L2 regularization (`lambda`) on weights.
- **Train/validation/test split:** Fixed split per run (e.g., 60/20/20 stratified in multi-run experiments, or explicit counts in phase scripts).
- **Number of runs and seed strategy:** Multi-seed runs (e.g., 3 seeds for champion comparison, 30 runs for statistical analysis). Seeds are deterministic and incremented per run.
- **Early stopping criteria:** Patience-based stopping on validation cost improvement; additional overfitting detection includes validation increase windows and relative gap thresholds.

## Mathematical Formulation
Let `x ∈ R^d` be the input, `W1, b1` hidden parameters, `W2, b2` output parameters.

**Feed-forward**
```
z1 = W1 x + b1
h  = tanh(z1)
z2 = W2 h + b2
ŷ  = softmax(z2)
```

**Loss (per sample)**
```
L = -∑_{k=1}^2 y_k log(ŷ_k) + (λ/2)(||W1||^2 + ||W2||^2)
```

**Objective**
Minimize the average cross-entropy with L2 regularization over the training set. Optimizers differ only in the parameter update rule; all other conditions are held constant.

## Optimizer Comparison Methodology
- **Fairness across optimizers:** Identical data splits, preprocessing (standardization using training statistics), initialization strategy, and stopping rules. Only the optimizer update rule is varied.
- **Deterministic evaluation protocol:** Fixed random seeds and controlled splits per run. Results are aggregated across seeds or 30-run trials for stability analysis.
- **Overfitting detection logic:** Validation cost increases tracked across a fixed window, and training/validation cost gaps are monitored. Runs flagged as overfit are excluded from champion selection or reported explicitly.

## Results Summary Table
Placeholder values are provided; replace with measured metrics from experiments.

| Optimizer | Mean Test Acc (%) | Std Dev (%) | Mean Val Cost | Convergence Stability |
|---|---:|---:|---:|---|
| GD  | 92.1 | 1.8 | 0.235 | Moderate |
| CG  | 93.4 | 1.2 | 0.210 | High |
| BFGS| 94.0 | 0.9 | 0.198 | High |
| DFP | 93.7 | 1.0 | 0.205 | High |
| ABC | 92.8 | 1.5 | 0.225 | Variable |

## Convergence Behavior Discussion
Gradient-based methods (BFGS, DFP, CG, GD) typically converge faster in terms of iteration count due to curvature-informed or conjugate direction updates. However, they are sensitive to learning rate and can exhibit sharper overfitting when validation cost rises quickly.  
Swarm-based optimization (ABC) explores the parameter space more broadly and can be less sensitive to local minima, but this often comes at the cost of slower convergence and higher variance across runs. The experiments therefore emphasize the trade-off between speed (second-order and conjugate methods) and stability (swarm exploration).

## Statistical Analysis Explanation
Multi-run evaluation reduces sensitivity to random initialization and data partitioning. The 30-run pipeline reports mean and standard deviation, bootstrap confidence intervals for mean test accuracy, and paired statistical tests (e.g., Wilcoxon signed-rank and paired t-tests) against the best-performing method. This improves validity by quantifying uncertainty and assessing whether observed gains are consistent across runs rather than artifacts of a single split.

## Methodological Contribution
The experiment isolates optimizer effects under a single, consistent pipeline while simultaneously evaluating overfitting, convergence stability, and statistical significance. This design is non-trivial because it requires aligning early stopping, initialization, and preprocessing across fundamentally different optimizers while preserving reproducibility and fairness in comparison.

## Setup Instructions
1. Install MATLAB (R2020b or later recommended).
2. Clone this repository and set MATLAB’s current folder to the repo root.
3. Ensure `apple_quality.csv` is present in the repository root.
4. Run a specific experiment script, for example:
   - `Hiperparametrelerin optimize olması/apple_ysa_phase6_champion_comparison.m`
   - `Modelin 1 run, 30 run, bias ve agırlık optimizasyonu/run_apple_ysa_30runs.m`
5. Generated figures are saved to a local `figures/` directory.

## Architecture Diagram
Placeholder: add a high-level pipeline diagram showing data flow, preprocessing, training, and evaluation.

## Developer Info
- **Author:** Metin Karyagdi
- **Affiliation:** Amasya University, Faculty of Engineering, Department of Computer Engineering
- **Location:** Amasya, Türkiye
