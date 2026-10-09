import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier

std_scaler = StandardScaler()
imputer = SimpleImputer(strategy="median")
imputer2 = SimpleImputer(strategy="median")
pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)

data = pd.read_csv('cs-training.csv')
data['RevolvingUtilizationOfUnsecuredLines'] = np.log1p(data["RevolvingUtilizationOfUnsecuredLines"])
data['RevolvingUtilizationOfUnsecuredLines'] = data['RevolvingUtilizationOfUnsecuredLines'].clip(upper=0.9163)
data['MonthlyIncome'] = np.log1p(data["MonthlyIncome"])
data.at[65695,'age'] = 52
data['DebtRatio'] = np.log1p(data["DebtRatio"])
data['DebtRatio'] = data['DebtRatio'].clip(upper=0.9163)
data = data.drop(columns=['Unnamed: 0'])
train_set, test_set = train_test_split(data, test_size=0.2, stratify= data['SeriousDlqin2yrs'],random_state=42)
train_set['MonthlyIncome'] = imputer.fit_transform(train_set[['MonthlyIncome']])
test_set['MonthlyIncome'] = imputer.transform(test_set[['MonthlyIncome']])
train_set['NumberOfDependents'] = imputer2.fit_transform(train_set[['NumberOfDependents']])
test_set['NumberOfDependents'] = imputer2.transform(test_set[['NumberOfDependents']])
predict_trinputs = train_set.drop('SeriousDlqin2yrs',axis=1)
predict_trlabels = train_set['SeriousDlqin2yrs'].copy()
predict_teinputs = test_set.drop('SeriousDlqin2yrs',axis=1)
predict_telabels = test_set['SeriousDlqin2yrs'].copy()
scaledtrinputs = std_scaler.fit_transform(predict_trinputs)
scaledteinputs = std_scaler.transform(predict_teinputs)
logreg= LogisticRegression(class_weight='balanced').fit(scaledtrinputs,predict_trlabels)
rfc = RandomForestClassifier(random_state=42).fit(predict_trinputs,predict_trlabels)
rfcb = RandomForestClassifier(class_weight='balanced',random_state=42).fit(predict_trinputs,predict_trlabels)
predict2 = rfc.predict(predict_teinputs)
predict3= rfcb.predict(predict_teinputs)
predict = logreg.predict(scaledteinputs)
threshold = 0.75
probs = rfc.predict_proba(predict_teinputs)
print(probs[:5])