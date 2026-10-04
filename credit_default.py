import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

imputer = SimpleImputer(strategy="median")
pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)

data = pd.read_csv('cs-training.csv')
data['RevolvingUtilizationOfUnsecuredLines'] = np.log1p(data["RevolvingUtilizationOfUnsecuredLines"])
data['RevolvingUtilizationOfUnsecuredLines'] = data['RevolvingUtilizationOfUnsecuredLines'].clip(upper=0.9163)
data['MonthlyIncome'] = imputer.fit_transform(data[['MonthlyIncome']])
data['MonthlyIncome'] = np.log1p(data["MonthlyIncome"])
data.at[65695,'age'] = 52
data['DebtRatio'] = np.log1p(data["DebtRatio"])
data['DebtRatio'] = data['DebtRatio'].clip(upper=0.9163)
data['NumberOfDependents'].hist(edgecolor= 'white', bins = 100, density=True)
data['NumberOfDependents'] = imputer.fit_transform(data[['NumberOfDependents']])
data = data.drop(columns=['Unnamed: 0'])
train_set, test_set = train_test_split(data, test_size=0.2, stratify= data['SeriousDlqin2yrs'],random_state=42)