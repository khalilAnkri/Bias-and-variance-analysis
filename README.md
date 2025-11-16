# ELEN0062 – Introduction to Machine Learning  
## Project 2 – Bias and Variance Analysis

This repository contains our implementation and report for **Project 2** of the ELEN0062 course at UCLouvain.  
The goal of the assignment is to study **bias–variance decomposition** both analytically and empirically using synthetic regression data.

---

## 📂 Repository Structure
ELEN0062-Project2/
│
├── report/
│ └── report.tex # LaTeX report (Part 1 + Part 2)
│
├── src/
│ ├── main.py # Main script running experiments
│ └── utils.py # Data generation + true function + helpers
│
└── figures/ # All generated plots for the report


---

## 🧪 What the Code Does

The Python scripts generate synthetic data according to the model:

- Features \(x_j \sim \mathcal{U}[-10,10]\)
- True function  
  \[
  h(x) = \sin(2x_1) + x_1 \cos(x_1 - 1)
  \]
- Noise \(\varepsilon \sim \mathcal{N}(0,1)\)

The experiments estimate **bias²**, **variance**, **residual error**, and **total error** for:

- **Ridge Regression**
- **k-Nearest Neighbors**
- **Regression Trees**
- **Bagging** applied to each method

The code also produces the plots required for:
- (2.2) Bias/variance as a function of \(x_1\)
- (2.3) Effect of training size, model complexity, irrelevant features
- (2.4) Effect of bagging

All figures are saved automatically in the `figures/` folder and included in the report.

---

## ▶️ Running the Project

1. Install dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

2. Run all experiments and generate figures:

```bash
python src/main.py
```

## 👥 Authors

Samira Ben Ahmed (s2503328)

Mohamed-Khalil Ankri (s2502523)

Ishahk Hamad (s2402246)