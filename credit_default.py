import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)

data = pd.read_csv('cs-training.csv')
data['RevolvingUtilizationOfUnsecuredLines'] = np.log1p(data["RevolvingUtilizationOfUnsecuredLines"])
data['RevolvingUtilizationOfUnsecuredLines'] = data['RevolvingUtilizationOfUnsecuredLines'].clip(upper=0.9163)
data.at[65695,'age'] = 52
data['DebtRatio'] = np.log1p(data["DebtRatio"])
data['DebtRatio'].hist(edgecolor= 'white', bins = 100, density=True)
data['DebtRatio'] = data['DebtRatio'].clip(upper=0.9163)
print((data['DebtRatio']>2.5).sum())
print((data['DebtRatio']>3).sum())
print((data['DebtRatio']>3.5).sum())
print((data['DebtRatio']>4).sum())
print((data['DebtRatio']>6).sum())


