# MLB Player Salary Prediction — Regression & Regularisation

A complete regression pipeline predicting Major League Baseball player salaries from career and season performance statistics. Five modelling approaches are benchmarked side-by-side — from ordinary least squares to Lasso — to identify the most generalisable model and the most predictive features.

## Highlights

- **Ridge regression achieves the best validation performance** (lowest MSE) among all five models tested
- **Lasso regression automatically selects a sparse feature subset** — zeroing out irrelevant predictors and retaining only the most predictive career statistics
- Polynomial regression (degree > 1) severely overfits: training R² = 0.98 but validation performance collapses — a clear demonstration of the bias-variance tradeoff
- Backward stepwise selection (AIC-based) converges on a parsimonious 10-feature model that outperforms the full 19-feature OLS model

## Project Structure

```
mlb-salary-prediction/
├── mlb_salary_prediction.ipynb    # Main analysis notebook
├── Hitters.csv                    # Dataset — 263 MLB players, 1986–87 seasons
└── README.md
```

## Analysis Sections

| Section | Description |
|---|---|
| 1. Data Loading & Split | Load Hitters dataset, 70/30 stratified train/validation split |
| 2. Linear Regression (OLS) | Full model with all 19 features — baseline |
| 3. Polynomial Regression | Degree 1–4 comparison, overfitting analysis |
| 4. Optimal Degree Selection | CV-based selection of polynomial degree (MSE minimisation) |
| 5. Stepwise Feature Selection | Backward & forward selection using AIC criterion |
| 6. Ridge Regression | L2 regularisation with cross-validated alpha (`RidgeCV`) |
| 7. Lasso Regression | L1 regularisation with cross-validated alpha (`LassoCV`) |
| 8. Model Comparison | Side-by-side MSE and R² for all five approaches |
| 9. Conclusions | Best model identification and feature importance insights |

## Model Results

| Model | Validation MSE | R² (Val) | Notes |
|---|---|---|---|
| Full OLS (19 features) | ~113,000 | ~0.48 | Baseline; some noisy features |
| Polynomial (degree 2) | High | ~0.10 | Severe overfitting |
| Backward Selection OLS | Lower than full OLS | — | Parsimonious 10-feature model |
| **Ridge Regression** | **Lowest** | **Best** | L2 shrinkage; optimal alpha via CV |
| Lasso Regression | Similar to Ridge | High | Sparse — zeros out irrelevant features |

> **Key takeaway:** Regularisation (Ridge/Lasso) and feature selection consistently outperform the full OLS model on unseen data — empirically demonstrating the bias-variance tradeoff.

## Dataset

The [Hitters dataset](https://www.statlearning.com/) from *An Introduction to Statistical Learning* (James et al.) contains statistics for 263 MLB players from the 1986–87 seasons, with 59 missing salary values (handled via listwise deletion).

**Features include:**
- Season stats: `AtBat`, `Hits`, `HmRun`, `Runs`, `RBI`, `Walks`
- Career stats: `CAtBat`, `CHits`, `CHmRun`, `CRuns`, `CRBI`, `CWalks`, `Years`
- Fielding: `PutOuts`, `Assists`, `Errors`
- Categorical: `League`, `Division`, `NewLeague`

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Statsmodels | OLS regression with detailed summaries |
| Scikit-learn | Polynomial features, Ridge/Lasso CV, metrics |
| Pandas | Data manipulation |
| Matplotlib | Actual vs. predicted visualisations |

## Getting Started

```bash
git clone https://github.com/himanshuladdhad/mlb-salary-prediction.git
cd mlb-salary-prediction
pip install pandas numpy matplotlib scikit-learn statsmodels jupyter
jupyter notebook mlb_salary_prediction.ipynb
```

> Ensure `Hitters.csv` is in the same directory as the notebook.

---

*Part of my data science portfolio. See also: [Telecom Churn EDA](https://github.com/himanshuladdhad/telecom-churn-eda) · [Bankruptcy Prediction Ensemble](https://github.com/himanshuladdhad/bankruptcy-prediction-ensemble)*
