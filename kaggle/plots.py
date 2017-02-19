import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
import pandas as pd
dataset = datasets.load_boston()
bos = pd.DataFrame(dataset['data'])


cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True, center="light")
sns.clustermap(bos.corr(), figsize=(10, 10), cmap=cmap)
