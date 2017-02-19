
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

#data2,target = dataset.data,dataset.target

data1 = pd.DataFrame(data= np.c_[boston['data'], boston['target']],
                     columns= boston['feature_names'] + ['target'])

bos = pd.DataFrame(dataset['data'])[0:12]
plt.matshow(bos)
features = pd.DataFrame(dataset['feature_names'])
x_pos = np.arange(len(features))
plt.xticks(x_pos,features)
y_pos = np.arange(len(features))
plt.yticks(y_pos,features)
print features
plt.show()
exit()
print bos.head()

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(bos)
ax.set_aspect('equal')

ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')

plt.show()
#plots.error_plot(bos)



fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(bos)


exit()
data = data_scaler.fit_transform(data)
target = target_scaler.fit_transform(target)

environment.reproducible()

x_train, x_test, y_train, y_test = train_test_split(
    data, target, train_size=0.85
)


cgnet = algorithms.ConjugateGradient(
    connection=[
        layers.Input(13),
        layers.Sigmoid(50),
        layers.Sigmoid(1),
    ],
    search_method='golden',
    show_epoch=25,
    verbose=True,
    addons=[algorithms.LinearSearch],
)

cgnet.train(x_train, y_train, x_test, y_test, epochs=100)

#plots.error_plot(cgnet)


#print data
