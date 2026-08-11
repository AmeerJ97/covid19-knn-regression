# Analyzing COVID-19 Search Trends and Hospitalizations

Course project for **McGill COMP 551: Applied Machine Learning — Mini-Project 1, Group 53, Fall 2020** (3-person team)

## Overview

This repository implements and evaluates two regression models that predict US COVID-19 hospitalizations from Google symptom search trends. It also includes an unsupervised exploration of the search-trend data using Principal Component Analysis (PCA) and K-Means clustering. The full write-up is available in the repository report: [Report/Comp 551 Project 1 Report G53.pdf](Report/Comp%20551%20Project%201%20Report%20G53.pdf).

## Datasets

Both datasets are sourced from the [Google Research Open COVID-19 Data](https://github.com/google-research/open-covid-19-data) repository:

1. **Symptom search trends (United States)** — daily relative popularity of symptom searches by region. After cleaning, the dataset contains **14,790 rows and 430 columns** (422 symptoms plus timestamps and region identifiers).
2. **Aggregated COVID-19 statistics** — public hospitalization, case, death, test, and discharge data for multiple regions. The raw dataset has **102,912 rows and 62 columns**.

The label used for supervised learning is `hospitalized_new`.

## Methods

### Preprocessing

The preprocessing pipeline:

- Removes regions with no non-zero hospitalization records.
- Drops columns with excessive missing values (NaN threshold: 95%).
- Collapses the daily search-trend data to weekly resolution.
- Merges search-trend and hospitalization data by region and date, keeping only the overlapping window **2020/03/02 to 2020/09/08**.
- Z-normalizes and Min-Max normalizes features, then fills remaining NaN values with the symptom-wise mean.

### Unsupervised Learning

- **PCA**: The symptom features are reduced with PCA. **80 principal components** are required to retain 95% of the variance, producing a reduced feature matrix of shape **(434, 81)**.
- **K-Means clustering**: Applied to the PCA-reduced data. The elbow method indicates an optimal cluster count of **K = 5**.

### Supervised Learning

- **K-Nearest Neighbours regression**: 5-fold cross-validation is used to select the hyperparameter. The best model uses **K = 12** and achieves a **5-fold cross-validation error of 0.0057**.
- **Decision Tree regression**: The best cross-validated model uses a maximum of **20 features** per split and achieves a **5-fold cross-validation error of 0.0133**.

## Results

| Model | Best hyperparameter | 5-fold CV error |
|-------|---------------------|-----------------|
| KNN regression | K = 12 | **0.0057** |
| Decision Tree regression | max features = 20 | **0.0133** |

## Calling Convention

The analysis is exposed through the `Loader` class in `code.ipynb`.

```python
loader = Loader()
dataset_ready = loader.get_merged_datasets()
loader.print_basic_stats()
loader.run_knn_crossval()
loader.run_decision_tree_crossval()
loader.run_knn_by_date()
loader.run_decision_tree_by_date()
loader.run_pca()
loader.run_Kmeans(k=5, kmax=10)
loader.plotAllRegionsForComparison(['Hawaii', 'Montana', 'Idaho', 'Maine', 'Wyoming'])
```

### Main `Loader` methods

- `get_merged_datasets()` — load, preprocess, and merge the two datasets.
- `plotAllRegionsForComparison(regions)` — plot symptom evolution over time for selected regions.
- `run_pca()` — perform Principal Component Analysis.
- `run_Kmeans(k, kmax)` — perform K-Means clustering.
- `run_knn_crossval()` / `run_knn_by_date()` — train and evaluate KNN regression by cross-validation and by date split.
- `run_decision_tree_crossval()` / `run_decision_tree_by_date()` — train and evaluate Decision Tree regression by cross-validation and by date split.

## Libraries

- NumPy
- Pandas
- scikit-learn
- Matplotlib
- pydotplus

## Report

For full methodology, figures, and discussion, see [Report/Comp 551 Project 1 Report G53.pdf](Report/Comp%20551%20Project%201%20Report%20G53.pdf).
