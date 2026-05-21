Online Shoppers Purchasing Intention

> Predicting e-commerce conversion using machine learning 


## Overview

This project predicts whether an online shopping session will result in a purchase using behavioral and session-level data from **12,330 Google Analytics user sessions**. Seven classification models are trained, tuned, and compared to identify the most effective approach for e-commerce conversion prediction.

**Best result:** Random Forest achieved **93.99% accuracy** and an **F1-score of 0.9407**, while XGBoost led on sensitivity at **94.50%** — catching the most potential buyers.


## Dataset

- **Source:** [UCI ML Repository — Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
- **Sessions:** 12,330 | **Features:** 18 | **Target:** `Revenue` (binary)
- **Class imbalance:** ~84% No Purchase / ~16% Purchase → addressed with **SMOTE**

| Variable | Type | Description |
|----------|------|-------------|
| `Administrative`, `Informational`, `ProductRelated` | Integer | Pages visited per category |
| `*_Duration` | Float | Time spent on each page category (seconds) |
| `BounceRates` | Float | % of visitors who leave after one page |
| `ExitRates` | Float | % of exits from a specific page |
| `PageValues` | Float | Avg. monetary value a page contributes to a session |
| `SpecialDay` | Float | Closeness of visit to a special day (0–1) |
| `Month`, `VisitorType` | Categorical | Session month; New / Returning / Other |
| `Weekend` | Boolean | Whether the session occurred on a weekend |
| `Revenue` | Boolean | **Target** — did the session result in a purchase? |

---

## Methodology

### Preprocessing & Feature Engineering
- Removed 125 duplicate rows
- `log1p` transformation on right-skewed duration and count features
- One-hot encoding for `Month` (10 levels) and `VisitorType` (3 levels)
- Z-score standardisation for distance-sensitive features (BounceRates, ExitRates, PageValues)
- **SMOTE** (K=5) applied after all transformations → balanced ~50/50 class split

### Feature Selection — PCA
- PCA run on 14 numeric features; Kaiser-Guttman criterion identified **5 principal components** (eigenvalue ≥ 1)
- First 6 PCs explain ~70% of total variance
- **PC1** driven by browsing engagement (ProductRelated, Administrative, Informational)
- **PC2** driven by BounceRates and ExitRates — capturing exit behaviour
- `PageValues`, `ExitRates`, `BounceRates`, and `ProductRelated_Duration` identified as top predictors
- Low-information features (`Region`, `Browser`, `OperatingSystems`, `TrafficType`) removed from model inputs

### Models — 70/30 Train/Test Split

| Model | Accuracy | Sensitivity | F1-Score |
|-------|----------|-------------|----------|
| **Random Forest** | **93.99%** | 93.92% | **94.07%** |
| XGBoost | 93.85% | **94.50%** | 93.97% |
| Decision Tree | 89.93% | 88.53% | 89.92% |
| KNN (k=5) | 87.21% | 85.20% | 87.11% |
| Logistic Regression | 83.09% | 88.87% | 84.21% |
| Naive Bayes | 71.63% | 56.18% | 66.76% |
| PCA Logistic Regression | 68.87% | 68.02% | 68.91% |

---

## Key Findings

- **`PageValues` is the single strongest predictor** of purchase intent — confirmed by both the Decision Tree (root node) and the Random Forest variable importance plot
- **Ensemble methods dominate** — Random Forest and XGBoost outperform all others by capturing non-linear interactions that linear models miss
- **November is the most purchase-predictive month** — `Month_Nov` ranks as the second most important feature in Random Forest
- **PCA hurt performance** — compressing variables into linear components removed the high-specificity signals (e.g. high PageValues in November) that drive e-commerce conversion
- **Logistic Regression missed 632 potential buyers** on the test set vs. only 182 for Random Forest — a meaningful difference in a real revenue context

---

## How to Run

**Requirements:** R with the following packages:

```r
install.packages(c("dplyr", "caret", "tree", "randomForest",
                   "e1071", "xgboost", "class", "rpart",
                   "rpart.plot", "smotefamily", "klaR", "readxl"))
```

**Steps:**

1. Clone the repository and set your working directory to the project folder
2. Place `online_shoppers_intention.csv` in that folder
3. Update the `setwd()` path at the top of `FinalProjectCode.txt`
4. Run the script top-to-bottom — each section is clearly labelled

The script runs all 7 models sequentially and prints a final comparison table of Accuracy, Sensitivity, and F1-Score.

---

## References

1. Suchacka et al. (2015) — SVM classification of e-customer sessions. *ECMS 2015.*
2. Suchacka et al. (2015) — KNN classification of user sessions. *JTIT.*
3. James, Witten, Hastie & Tibshirani (2021) — *An Introduction to Statistical Learning* (2nd ed.). Springer.
4. Mobasher et al. (2001) — Aggregate usage profiles for web personalisation. *Data Mining and Knowledge Discovery.*
5. Moe (2003) — Differentiating online shoppers using clickstream. *Journal of Consumer Psychology.*
6. Swastik (2025) — SMOTE for imbalanced classification. *Analytics Vidhya.*
