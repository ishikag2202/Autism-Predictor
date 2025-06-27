# Autism-Predictor

## Introduction
Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition that affects how a person communicates, interacts, and behaves. Early detection plays a critical role in providing timely support and improving long-term outcomes.

This project is a machine learning-based web application built with Streamlit that predicts the likelihood of ASD in individuals using a simple screening questionnaire and personal information. The app leverages supervised learning models—Decision Tree, Random Forest, and XGBoost-trained on Kaggle dataset. Among these, Random Forest showed the highest accuracy and was chosen for final predictions.

The application offers a simple, accessible interface where users can enter inputs, receive instant predictions, and even generate PDF reports. It highlights how technology can assist in creating accessible, early-stage screening solutions for health-related assessments.

## Tech Stack Used
-  Python - Core programming language used to implement data processing, model training, and the backend logic of the web app.
-  Streamlit - Used to build the interactive web interface. It handles user input forms, layout, prediction display, and report downloads—all in a browser-based app.
-  pandas - Utilized for reading and manipulating structured data (e.g., screening dataset, input handling, formatting).
-  NumPy - Used for numerical operations and handling array-based data required by machine learning models. 
-  scikit-learn - Provided ML algorithms like Decision Tree and Random Forest, as well as data preprocessing and model evaluation utilities.
-  XGBoost - Integrated as an additional high-performance model to improve accuracy and compare results with other classifiers.
-  matplotlib - Used to visualize model comparisons or prediction outputs if graphical display is added (e.g., pie charts, bar plots).
-  FPDF - Used to generate downloadable PDF reports summarizing the prediction results and user inputs.
-  pickle - Used to serialize and load the trained machine learning model (model.pkl) so it can be reused during prediction without retraining.

## Machine Learning Models
This project implements and compares three popular supervised machine learning models for binary classification (Autistic / Not Autistic):
#### 1. Decision Tree Classifier:
  It is a tree-structured model that splits the data into branches based on feature thresholds, making decisions by following a set of if-else conditions.     It works by recursively partitioning the dataset into subsets that increase homogeneity with respect to the target variable. Due to its simplicity and       interpretability, it’s often used as a baseline model in classification problems.
  In this project, the Decision Tree Classifier was implemented to provide an initial understanding of how the features affect the ASD prediction. While it    offered good interpretability, it tended to overfit on training data, leading to lower generalization accuracy. Therefore, it was mainly used as a           reference for evaluating more robust models.




