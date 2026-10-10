import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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

probathreshold_crossval = cross_val_predict(impute_forestclassifier_pipeline,train_inputs,train_labels,cv=5,n_jobs= -1,method= "predict_proba")
default_probs = probathreshold_crossval[:,1]
flagged = (default_probs>0.175).astype(int)
print(confusion_matrix(train_labels,flagged))
print(recall_score(train_labels,flagged))