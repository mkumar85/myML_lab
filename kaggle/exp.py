from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from neupy import environment
from neupy import plots
from neupy import algorithms, layers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()

boston = datasets.load_boston()
fig = plt.figure(figsize=(30,10))
data1 = pd.DataFrame(data= np.c_[boston['data'], boston['target']],dtype=float)
ax= fig.add_subplot(111)
cax = ax.matshow(data1.corr(),vmin=-1,vmax=1)
fig.colorbar(cax)
bos = pd.DataFrame(boston['data'])[0:12]
#plt.matshow(bos)
features = pd.DataFrame(boston['feature_names'])
x_pos = np.arange(len(features))
plt.xticks(x_pos,features[0])
y_pos = np.arange(len(features))
plt.yticks(y_pos,features[0])

plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Boston housing Dataset Correlations')

print features
plt.show()

