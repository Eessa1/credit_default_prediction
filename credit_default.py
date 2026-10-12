import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

run_sweep = False #true reruns the threshold sweep

pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)
impute_forestclassifier_pipeline = Pipeline([("impute",SimpleImputer(strategy="median")),("rfc", RandomForestClassifier(random_state=42))])

data = pd.read_csv('cs-training.csv')
data['RevolvingUtilizationOfUnsecuredLines'] = np.log1p(data["RevolvingUtilizationOfUnsecuredLines"])
data['RevolvingUtilizationOfUnsecuredLines'] = data['RevolvingUtilizationOfUnsecuredLines'].clip(upper=0.9163)
data['MonthlyIncome'] = np.log1p(data["MonthlyIncome"])
data.at[65695,'age'] = 52
data['DebtRatio'] = np.log1p(data["DebtRatio"])
data['DebtRatio'] = data['DebtRatio'].clip(upper=0.9163)
data = data.drop(columns=['Unnamed: 0'])

train_set, test_set = train_test_split(data, test_size=0.2, stratify= data['SeriousDlqin2yrs'],random_state=42)
train_inputs = train_set.drop('SeriousDlqin2yrs',axis=1)
train_labels = train_set['SeriousDlqin2yrs'].copy()
test_inputs = test_set.drop('SeriousDlqin2yrs',axis=1)
test_labels = test_set['SeriousDlqin2yrs'].copy()

impute_forestclassifier_pipeline.fit(train_inputs,train_labels)
testpredict = impute_forestclassifier_pipeline.predict_proba(test_inputs)
test_probs = testpredict[:,1]
test_flags = (test_probs>0.07).astype(int)
print(confusion_matrix(test_labels,test_flags))
print(precision_score(test_labels,test_flags))
print(recall_score(test_labels,test_flags))

if run_sweep:
    oof_probs = cross_val_predict(impute_forestclassifier_pipeline,train_inputs,train_labels,cv=5,n_jobs= -1,method= "predict_proba")
    default_probs = oof_probs[:,1]
    thresholds = np.round(np.arange(0.01,0.51,0.01),2)
    costs = []
    c= 5
    for t in thresholds:
        flagged = (default_probs>t).astype(int)
        tn,fp,fn,tp= confusion_matrix(train_labels,flagged).ravel()
        cost = (c*fn)+fp
        costs.append(cost)

    bestmin = np.argmin(costs)
    print(thresholds[bestmin])
    print(costs[bestmin])
    flagcheck= (default_probs>thresholds[bestmin]).astype(int)
    print(confusion_matrix(train_labels,flagcheck))
    print(precision_score(train_labels,flagcheck))
    print(recall_score(train_labels,flagcheck))
        

