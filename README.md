# credit_default_prediction

## Problem
Predicting whether a borrower will experience serious delinquency (90+ days past due)

## Data
- Source: Kaggle "Give Me Some Credit" competition (cs-training.csv)
- 150,000 rows, 10 features + target  (SeriousDlqin2yrs)
- Significant class imbalance: only 6.7% of borrowers defaulted, accuracy alone is misleading

## Approach
1. Cleaned the data: log transformed and clipped RevolvingUtilization and DebtRatio, fixed an invalid age of 0, log transformed MonthlyIncome and filled missing income and dependents with the median
2. Split the data 80/20 (stratified), keeping the test set untouched until the end
3. Built a baseline logistic regression model
4. Compared against a random forest
5. Put the imputer and forest into one pipeline, so cross validation refits the imputer on each fold's training portion without a leak
6. Chose decision thresholds from out of folds probabilities by minimising misclassification cost
7. Evaluated using precision,recall and a confusion matrix rather than accuracy alone given the class imbalance

## Results
I compared models on recall for defaulters (90+ days past due), because only 6.7% of applicants default and plain accuracy is misleading (first logistic regression had 93% accuracy with around 5% recall). At the default 0.5 cutoff the random forest caught only 19% of defaulters and balancing the class weights raised that to 37% so the next thing to tune was the threshold

To choose a decision threshold without using the test set, I used 5 fold cross validation on the training data, with the imputer inside a pipeline to refit on each folds training portion. For each cost ratio c (one missed defaulter costs the same as c false alarms), I picked the threshold with the lowest total cost, then evaluated on the test set once.

| c | Threshold |Recall(CV/test)| Precision(CV/test) | False alarms per 1000 applicants (CV/test) |
| :---: | :---: | :---:  | :---: | :---: |
| 5 | 0.21  | 50.7%/51.5% | 34.8% / 35.1% | 63/64 |
| 10| 0.11  | 65.5%/66.7% | 25.2% / 25.9% | 130/128 |
| 16| 0.07  | 73.7%/74.9% | 20.4% / 20.6% | 193/193 |

CV = out of fold predictions on the training set. 
Test = test set predictions

Test results matched the cross validated ones to within about a point, so the thresholds carried over to unseen data. The right row depends on what a missed defaulter costs the bank compared to a wrongly flagged customer, which I don't know. 75% recall only becomes the cheapest choice when a miss costs around 16 false alarms. At that point the model catches about 50 of the 67 defaulters per 1000 applicants, at the price of 193 false alarms.

For comparison, the balanced logistic regression reached 75% recall with 18% precision and 7056 false alarms on the test set. The forest at c= 16 reached 75% recall with 20.6% precision and 5777 false alarms, 1279 less.

## What I'd do differently
- Tune the logistic regressions threshold the same way. At the moment the 75% recall is where the default cutoff landed so the comparison isn't fully fair
- Get real cost figures. The cost ratio c is an assumption, not something I could measure, a real cost ratio would let me choose a single threshold instead of a range of options
- Use repeated or different splits. Everything comes from one split, and the test set has only 2005 defaulters
- Clipping RevolvingUtilization and DebtRatio at 1.5 was plausible, no clear elbow in the data. I'd test how sensitive the results are to that cutoff.
- Try other models and tune hyperparameters e.g. XGBoost

## How to run
Download cs-training.csv from Kaggle (give me some credit competition) and put it in the project folder, run credit_default.py. By default it trains the final model and evaluates on the test set at a threshold of 0.07. To rerun the threshold search set run_sweep = true. Change c in the script to test a different cost ratio and change threshold in the test flags line to evaluate the result on the test set