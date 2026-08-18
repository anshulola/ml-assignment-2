## ML Assignment 2

Name: Anshul Ola
Roll Number: 2025ac05001

### a. Problem statement

The goal of this assignment is to build and compare multiple classification models on a single dataset, then wrap them in an interactive Streamlit app so predictions and evaluation metrics can be viewed from a browser. This project treats breast tumor diagnosis as the classification task, predicting whether a tumor is malignant or benign from a set of measured cell features.

### b. Dataset description

Dataset: Breast Cancer Wisconsin (Diagnostic), sourced from the UCI Machine Learning Repository (also shipped through scikit-learn as `load_breast_cancer`).

- Instances: 569
- Features: 30 numeric features computed from digitized images of fine needle aspirate cell nuclei (radius, texture, perimeter, area, smoothness, and similar measurements, each reported as mean, standard error, and worst value)
- Target: binary, 0 for malignant and 1 for benign
- Class balance: roughly 63 percent benign and 37 percent malignant

An 80/20 stratified split was used. The 20 percent test split (114 rows) is saved as `test_data.csv` in this repo and is what the Streamlit app uses by default.

### c. GitHub Repository Link

https://github.com/anshulola/ml-assignment-2

### d. Models used

All 5 models were trained on the same train split and evaluated on the same held out test split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

Observations on model performance:

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer here. The 30 features are fairly linearly separable after scaling, so a simple linear decision boundary already gets almost everything right, and it stayed the top or near top model on every metric. |
| Decision Tree | Weakest of the five. A single tree with limited depth overfits some patterns in the training split and misses more malignant cases than the other models, which shows up as the lowest recall, F1 and MCC. |
| kNN | Strong performer, and it got every single benign and malignant case with a positive label right in terms of recall (1.0 on this split). Being distance based, it benefits a lot from the feature scaling done before training. |
| Naive Bayes | Decent but noticeably behind the top models. Gaussian Naive Bayes assumes the features are independent given the class, which is not really true for measurements like radius, perimeter and area that are correlated with each other, so accuracy takes a small hit. |
| Random Forest (Ensemble) | Solid, consistent scores across the board, close to but not quite matching Logistic Regression here. Averaging many trees fixes most of the single Decision Tree's overfitting problem, though it does not fully catch up to the simpler linear model on this particular dataset. |
| Overall Winner for your dataset? | Logistic Regression, it leads on accuracy, AUC, precision, F1 and MCC, and is only edged out by kNN on recall alone. |

### Repository structure

```
ml-assignment-2/
  app.py
  requirements.txt
  README.md
  test_data.csv
  .python-version
  model/
    train.py
    logistic_regression.pkl
    decision_tree.pkl
    knn.pkl
    naive_bayes.pkl
    random_forest.pkl
    metrics.csv
    feature_names.json
```

### How to run locally

```
pip install -r requirements.txt
python model/train.py
streamlit run app.py
```

### Live Streamlit app

https://ml-assignment-2-x2hflsyvt6kotlozznwba9.streamlit.app/

### Streamlit app features

- CSV upload for test data, matching the schema of `test_data.csv`
- Dropdown to pick which of the 5 trained models to run
- Accuracy, AUC, precision, recall, F1 and MCC shown for the selected model
- Confusion matrix and a full classification report
- A side by side comparison table and bar chart across all 5 models on the same uploaded test data
