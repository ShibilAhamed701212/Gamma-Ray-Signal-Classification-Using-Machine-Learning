# Gamma-Ray Signal Classification Using Machine Learning

A machine learning project to classify gamma-ray signals from cosmic-ray background using data from the **MAGIC Gamma Telescope** (UCI ML Repository).

## Overview

Cherenkov gamma telescopes detect high-energy gamma rays by capturing Cherenkov radiation from electromagnetic showers in the atmosphere. The challenge is to **distinguish genuine gamma-ray events (signal) from hadronic showers caused by cosmic rays (background)** based on the shape and characteristics of the shower images.

## Dataset

- **Source:** [UCI ML Repository — MAGIC Gamma Telescope](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)
- **Instances:** 19,020
- **Features:** 10 continuous Hillas parameters (`fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`)
- **Classes:** Gamma (signal) — 12,332 | Hadron (background) — 6,688

## Models & Results

| Model | Accuracy |
|---|---|
| K-Nearest Neighbors (k=5) | 82% |
| Naive Bayes | 72% |
| Logistic Regression | 77% |
| Support Vector Machine | 87% |
| Neural Network (TensorFlow) | Best selected via hyperparameter tuning |

**Preprocessing:** StandardScaler normalization + RandomOverSampler for class imbalance  
**Split:** 60% train / 20% validation / 20% test

## Tech Stack

- Python 3
- NumPy, Pandas, Matplotlib
- Scikit-learn, Imbalanced-learn
- TensorFlow / Keras

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ShibilAhamed701212/Gamma-Ray-Signal-Classification-Using-Machine-Learning.git

# Install dependencies
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow

# Open the notebook
jupyter notebook Copy_of_fcc_MAGIC_example.ipynb
```

## License

This project uses the MAGIC Gamma Telescope dataset from the UCI Machine Learning Repository.
