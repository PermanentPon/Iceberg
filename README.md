# Iceberg
This is my solution for [Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge).

# Result
Public LB - 323rd(top 10%), bronze medal

# Structure
* trainer.py - data wrangling and NN training
* predictor.py - predict classes based on test data using NNs
* trees.py - train LightGBM and xgboost to diversify results from NNs
* xgbsearch.py - baysian hyperparameter optimization
* models - custom models
* utils - custom transformation, logger and train/predict utils

# How to use
Run trainer.py to train neural networks. Run trees.py to train and blend LightGBM and XGBoost trees
