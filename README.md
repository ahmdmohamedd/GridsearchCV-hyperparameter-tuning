# **Hyperparameter Tuning for Machine Learning Models**

This project demonstrates hyperparameter tuning using `GridSearchCV` for three different machine learning models: **Random Forest**, **Support Vector Machines (SVM)**, and **K-Nearest Neighbors (KNN)**. The tuning is performed using the **F1 score** as the evaluation metric. Each model is implemented in a separate Jupyter notebook.

## **Table of Contents**
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Notebooks Overview](#notebooks-overview)
  - [1. Random Forest Tuning](#1-random-forest-tuning)
  - [2. SVM Tuning](#2-svm-tuning)
  - [3. KNN Tuning](#3-knn-tuning)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)

---

## **Project Overview**

Hyperparameter tuning is a critical step in building efficient machine learning models. This project explores the use of `GridSearchCV` for hyperparameter tuning of three different models:
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

The goal is to find the optimal hyperparameters for each model using the **F1 score** as the primary evaluation metric. The tuned models are then evaluated to measure their performance on unseen data.

---

## **Installation**

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/GridsearchCV-hyperparameter-tuning.git
   cd GridsearchCV-hyperparameter-tuning
   ```

2. Ensure that you have **Jupyter Notebook** or **JupyterLab** installed. You can install it via pip:
   ```bash
   pip install notebook
   ```

---

## **Dataset**

This project uses the **Iris dataset** for demonstration purposes. The dataset is included in the `scikit-learn` library. If you'd like to use a different dataset, you can replace the dataset loading part in the notebooks with your own dataset.

- **Iris dataset**: A simple, widely used dataset for classification. It contains 150 samples from each of three species of Iris flowers with four features: sepal length, sepal width, petal length, and petal width.

---

## **Notebooks Overview**

### 1. **Random Forest Tuning**
- **File**: `Randomforest_tunning.ipynb`
- **Objective**: Tune hyperparameters of the **Random Forest Classifier** using the F1 score.
- **Hyperparameters tuned**:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of the trees.
  - `min_samples_split`: Minimum number of samples required to split an internal node.
  - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
  
### 2. **SVM Tuning**
- **File**: `svm_tuning.ipynb`
- **Objective**: Tune hyperparameters of **Support Vector Machines (SVM)** using the F1 score.
- **Hyperparameters tuned**:
  - `C`: Regularization parameter.
  - `kernel`: Kernel type (linear, rbf, etc.).
  - `gamma`: Kernel coefficient.
  - `degree`: Degree for polynomial kernels.
  
### 3. **KNN Tuning**
- **File**: `KNN_tunning.ipynb`
- **Objective**: Tune hyperparameters of the **K-Nearest Neighbors (KNN)** classifier using the F1 score.
- **Hyperparameters tuned**:
  - `n_neighbors`: Number of neighbors to use.
  - `weights`: Weight function used in prediction.
  - `metric`: Distance metric to use for tree.
  
---

## **Results**

- **Best Model for Each Algorithm**: After tuning the hyperparameters using `GridSearchCV`, the best model for each algorithm is selected based on the highest **F1 score**.
- **F1 Score**: The F1 score was chosen because it balances precision and recall, which is particularly useful when the classes are imbalanced.

| Model           | Best Hyperparameters                                   | Best F1 Score (CV) |
|-----------------|--------------------------------------------------------|--------------------|
| Random Forest   | 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50 | 0.94               |
| SVM             | 'C': 1, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'                 | 0.97               |
| KNN             | 'metric: euclidean', 'n_neighbors: 7', 'weights: uniform' | 0.95              |

---

## **Usage**

To reproduce the results or run the tuning on a different dataset:
1. Open the notebook for the desired model in Jupyter Notebook:
   ```bash
   jupyter notebook Randomforest_tunning.ipynb
   ```
2. Replace the dataset loading code with your dataset if necessary.
3. Run the notebook cells sequentially to perform hyperparameter tuning.

---

## **Future Work**

- Experiment with more advanced models like **XGBoost**, **LightGBM**, and **Neural Networks**.
- Perform hyperparameter tuning using more sophisticated methods like **RandomizedSearchCV** or **Bayesian Optimization**.
- Explore different scoring metrics such as **ROC-AUC**, **Precision-Recall**, or **Log Loss**.

---

## **Contributing**

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your_username/hyperparameter-tuning/issues).

---
