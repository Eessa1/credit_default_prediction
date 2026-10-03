# credit_default_prediction

## Problem
Predicting whether a borrower will experience serious delinquency (90+ days past due)

## Data
- Source: Kaggle "Give Me Some Credit" competition (cs-training.csv)
- 150,000 rows, 11 features + target  (SeriousDlqin2yrs)
- Significant class imbalance: only ~6.7% of borrowers defaulted, accuracy alone is misleading

## Approach
1. Cleaned the data - handled extreme outiers, missing values and invalid entries across multiple columns
2. Built a baseline logistic regression model
3. Compared against a stronger model (XGBoost, random forest)
4. Evaluated using precision,recall and confusion matrix rather than accuracy alone given the class imbalance

## Results
(tbd)

## What I'd do differently
(tbd)