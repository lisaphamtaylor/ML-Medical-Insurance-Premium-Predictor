# What's My Premium - a health insurance premium prediction model

## Table of Contents
  - [Introduction](#introduction)
    - [Project Status](#project-status)
  - [Roadmap](#roadmap)
  - [License](#license)


## Introduction

What's My Premium is a health insurance premium prediction model that uses machine learning to predict the cost of health insurance based on a number of factors. This code compares the cross value score and train/test accuracies of different ML models--Linear Regression, Support Vector Regression, Random Forest, Gradient Boost, XGBoost--to identify the best model for prediction. The models are trained on a Kaggle dataset of 1338 entries. 

Dataset can be found [here](https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction/data).
Check out the original setup inspiration [here](https://www.geeksforgeeks.org/medical-insurance-price-prediction-using-machine-learning-python/).

### Project Status
11/10/2023: currently under development

## Roadmap
- [x] exploratory data analysis (EDA) with dataset 
- [x] data preprocessing
- [x] train and validate model for best random state number
- [x] linear regression model
- [x] support vector regression model
- [x] random forest model
- [x] gradient boosting model
- [x] XGBoost model
- [ ] integrate with the [Federal Marketplace API](https://marketplaceapicms.docs.apiary.io/#introduction/about)
- [ ] create web app front
- [ ] deploy to webpage

## License

This project is licensed under the MIT License.
