import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

pd.set_option('display.max_columns', None)
pd.set_option('display.width',None)

data = pd.read_csv('cs-training.csv')
data['RevolvingUtilizationOfUnsecuredLines'] = np.log1p(data["RevolvingUtilizationOfUnsecuredLines"])
data['RevolvingUtilizationOfUnsecuredLines'] = data['RevolvingUtilizationOfUnsecuredLines'].clip(upper=0.9167)
