import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)

data = pd.read_csv('cs-training.csv')
log_ruoul = np.log1p(data["RevolvingUtilizationOfUnsecuredLines"])
plt.hist(log_ruoul,density=True,bins=100)

replaced = data
