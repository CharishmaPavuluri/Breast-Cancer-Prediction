# Breast-Cancer-Prediction

## Introduction

Breast cancer is a critical health concern, and early detection plays a pivotal role in improving patient outcomes. This project aims to develop and compare machine learning models for breast cancer prediction, with a primary focus on identifying the most effective and robust model for early detection.

## Objective

The primary objective of this project is to leverage machine learning techniques to enhance breast cancer prediction. The project specifically explores the effectiveness of various models, including logistic regression (as a baseline), XGBoost, Random Forest, and Convolutional Neural Network (CNN). The goal is to identify a model that can provide accurate and early predictions, contributing to more effective treatments.

## Models

### 1. Logistic Regression (Baseline)

- **Purpose:** Logistic regression serves as the baseline model for binary classification.
- **Advantages:** Interpretability, simplicity, and computational efficiency.
- **Implementation:** [Link to Logistic Regression Code]

### 2. XGBoost

- **Purpose:** XGBoost is an ensemble learning algorithm known for its robustness and high performance.
- **Advantages:** Ensemble learning, robustness, and handling of missing data.
- **Implementation:** [Link to XGBoost Code]

### 3. Random Forest

- **Purpose:** Random Forest, an ensemble learning method, is explored for its ability to handle complex relationships and feature importance ranking.
- **Advantages:** Ensemble learning, feature importance, and reduced overfitting.
- **Implementation:** [Link to Random Forest Code]

### 4. Convolutional Neural Network (CNN)

- **Purpose:** CNN is applied for breast cancer prediction, particularly when working with image data.
- **Advantages:** Image processing, hierarchical feature learning, and suitability for complex data.
- **Implementation:** [Link to CNN Code]

## Dataset

- [Provide information about the dataset used, including the source, features, and any preprocessing steps.]

## Evaluation Metrics

- [Specify the evaluation metrics used to assess the performance of each model, such as accuracy, precision, recall, and area under the ROC curve.]

## Results

- [Present and discuss the results obtained from each model, highlighting the strengths and weaknesses of each approach.]

## Usage

- [Include instructions on how to use and run the code, along with any dependencies.]

## Future Improvements

- [Discuss potential enhancements, additional features, or avenues for future research.]

## Contributors

- [List individuals or organizations involved in the project.]

## License

- [Specify the license under which the project code is distributed.]

## Acknowledgments

- [Express gratitude or acknowledgement to any external resources, datasets, or contributors.]

## importing required libraries 

## Overview

This project highlights the crucial role of various libraries in different stages of the data science and machine learning workflow. The workflow is divided into key components: Data Handling, Visualization, Preprocessing and Feature Selection, and Machine Learning Model Development.

## Key Components

### 1. Data Handling Libraries

Data handling libraries lay the foundation for successful analysis and model development. They provide essential tools for loading, manipulating, and organizing data efficiently. In this project, we leverage the following library:

- **Pandas:** A powerful data manipulation library that facilitates data cleaning, filtering, and transformation.

### 2. Visualization Libraries

Visualization libraries empower us to create clear and informative visual representations, aiding in the exploration, interpretation, and communication of patterns and insights within the data. The project utilizes:

- **Matplotlib:** A versatile plotting library for creating static, interactive, and publication-quality visualizations.
- **Seaborn:** A high-level interface for statistical graphics, built on top of Matplotlib.

### 3. Preprocessing and Feature Selection Libraries

Preprocessing and feature selection libraries are crucial for transforming raw data into a format suitable for modeling. They enhance model performance by selecting the most relevant features and improving interpretability. The project incorporates:

- **Scikit-learn:** A comprehensive machine learning library that includes modules for preprocessing data, feature scaling, and feature selection.
- **Feature-Engine:** A library designed for feature engineering tasks, addressing challenges such as missing data and categorical variable encoding.

### 4. Machine Learning Model Libraries

Machine learning model libraries facilitate the development of accurate and robust classification models. The project leverages:

- **Scikit-learn:** A versatile library for classical machine learning algorithms, providing consistent interfaces for model training, evaluation, and hyperparameter tuning.
- **XGBoost:** An efficient implementation of gradient boosting, known for its scalability and performance in classification tasks.

## Dataset

The dataset used in this project was collected from Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/imtkaggleteam/breast-cancer/). It consists of 569 rows and 33 columns.

### Dataset Information

- **Rows:** 569
- **Columns:** 33

The target attribute in the dataset is "diagnosis."

# Breast Cancer Prediction Project

## Overview

This project focuses on developing and comparing machine learning models for the early detection of breast cancer. The workflow encompasses key components such as Data Handling, Visualization, Preprocessing and Feature Selection, and Machine Learning Model Development.

## Dataset

The dataset used in this project was collected from Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/imtkaggleteam/breast-cancer/). It consists of 569 rows and 33 columns.

### Dataset Information

- **Rows:** 569
- **Columns:** 33

The target attribute in the dataset is "diagnosis."

## Data Preprocessing

Data preprocessing involved several steps to ensure the dataset was clean and suitable for modeling:

### 1. Handling Missing Values

Only one column was found to have missing values. Since all entries in this column had missing values, the column was dropped. As a result, the dataset became clean with no missing values.

### 2. Handling Duplicates

Duplicates were checked, and the dataset was found to have none.

### 3. Handling Outliers

A good number of columns were found to have outliers. The identified outliers were capped, and the resulting dataframe had none.

### 4. Data Encoding

All categorical attributes were encoded using Label Encoder to prepare the attributes for the machine learning models.

# Exploratory Data Analysis

### Statistical Description

Statistical description provides a comprehensive overview of the dataset's key metrics for various breast cancer features, including count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum values.

### Correlation Analysis

Correlation analysis was performed to understand the relationships between different features in the dataset. Here are notable correlations:

- **Strong Positive Correlation:**
  - Between 'radius_mean' and 'area_mean' (0.997692)
  - Indicates a high degree of linear association.

- **Weak Correlation:**
  - Between 'id' and 'diagnosis' (0.065270)
  - Suggests a limited linear relationship.

## Feature Selection

### Recursive Feature Elimination (RFE)

RFE was used to select the n best-performing features.

## Model Training and Evaluation

### Data Splitting

The data was split into training and test sets, with a test size of 20% and a random state of 42.

### Logistic Regression Model

A Logistic Regression model was trained and tested using the training and test sets. It achieved the following performance metrics:

- **Accuracy:** 0.97
- **Precision:** 0.976
- **Recall:** 0.95
- **F1 Score:** 0.96

- ### XGBoost Model

An XGBoost model was trained and tested using the training and test sets. It achieved the following performance metrics:

- **Accuracy:** 0.96
- **Precision:** 0.95
- **Recall:** 0.95
- **F1 Score:** 0.95

- ### Random Forest Classifier Model

A Random Forest Classifier model was trained and tested using the training and test sets. It achieved the following performance metrics:

- **Accuracy:** 0.956
- **Precision:** 0.95
- **Recall:** 0.93
- **F1 Score:** 0.94

### Convolutional Neural Network (CNN) Model

A CNN model was trained and tested using the training and test sets. It achieved the following performance metrics:

- **Accuracy:** 0.956
- **Precision:** 0.975
- **Recall:** 0.907
- **F1 Score:** 0.94

- 
