import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)

data = pd.read_csv('cs-training.csv')
scaledincome = preprocessing.StandardScaler().fit_transform(data["MonthlyIncome"])
plt.hist(scaledincome,density=True,bins=30)
plt.show()