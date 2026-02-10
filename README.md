ML Assignment 2 – Breast Cancer Classification

a. Problem Statement

The objective of this project is to implement and compare six different machine learning classification models on a binary classification problem. The goal is to evaluate the performance of each model using multiple evaluation metrics and deploy the best-performing models through a Streamlit web application.

The classification task is to predict whether a tumor is malignant (0) or benign (1) based on diagnostic medical features.

b. Dataset Description

Dataset Name: Breast Cancer Wisconsin (Diagnostic)
Source: UCI Machine Learning Repository
Number of Instances: 569
Number of Features: 30 numerical features
Target Classes:

0 → Malignant

1 → Benign

c. Models Used

The following six machine learning models were implemented:

Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble)

Each model was trained using an 80-20 train-test split with stratified sampling

Model Comparison Table :

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree            | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| KNN                      | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes              | 0.9386   | 0.9878 | 0.9452    | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561   | 0.9931 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble)       | 0.9561   | 0.9950 | 0.9467    | 0.9861 | 0.9660 | 0.9058 |


| ML Model Name                | Observation about model performance                                                                                                                                                                                                                                     
| **Logistic Regression**      | Achieved the highest overall performance with the best Accuracy (98.24%), AUC (0.9954), and MCC (0.9623). The 									strong results indicate that the dataset is close to linearly separable. Logistic Regression demonstrated 									excellent balance between precision and recall. |

| **Decision Tree**            | Performed the weakest among all models with lower Accuracy (91.23%) and MCC (0.8174). Likely overfitted to training 									data and did not generalize as well as ensemble methods. |

| **KNN**                      | Delivered strong performance (95.61% Accuracy) with high recall. Distance-based classification worked well due to 									normalized numeric features, but performance was slightly below Logistic Regression.  |

| **Naive Bayes**              | Achieved good recall and competitive AUC (0.9878). However, independence assumptions between features may limit 								performance compared to more flexible model  |

| **Random Forest (Ensemble)** | Showed robust and stable performance (95.61% Accuracy, AUC 0.9931). Ensemble averaging reduced overfitting compared								 to Decision Tree and improved generalization  |

| **XGBoost (Ensemble)**       | Performed comparably to Random Forest with very high AUC (0.9950). Gradient boosting captured complex relationships									 effectively, though it did not surpass Logistic Regression on this dataset  |
