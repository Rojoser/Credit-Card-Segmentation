#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:45:59 2022

@author: jvk
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.manifold import locally_linear_embedding
from sklearn.decomposition import PCA
import copy
from scipy.stats import shapiro, anderson


path="/home/jvk/Desktop/propulsion/ML_Group_Challenge/Credit-Card-Segmentation/data/card_transactions.csv"

card_df = pd.read_csv(path).fillna(0)
myfeats=card_df.columns.to_list()
print(myfeats)

categorical_features = card_df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_features = card_df.select_dtypes(include=['int', 'float']).columns.tolist()

std=StandardScaler()
pca = PCA()
transf_df=copy.copy(card_df[numeric_features])
transf_df=std.fit_transform(transf_df)
print(type(transf_df))
X_pca = pca.fit_transform(transf_df)

print(transf_df.shape)
print(X_pca.shape)

locquals=[]
for varind in range(len(X_pca.T)):
    var=X_pca.T[varind]
    shpro=shapiro(var)
    # print(shpro)
    andsn=anderson(var)
    # print(andsn)
    # locquals.append(np.std(var)*shpro.statistic)
    locquals.append(andsn.statistic)
    # locquals.append(np.std(var)*andsn.statistic)

plotindx1=0
plotindx2=0

for locqualIND in range(len(locquals)):
    if locquals[locqualIND]==sorted(locquals)[-1]:
        plotindx1=locqualIND
    if locquals[locqualIND]==sorted(locquals)[-2]:
        plotindx2=locqualIND
print(plotindx1)
print(plotindx2)
# plotindx1=0
# plotindx2=1

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(X_pca[:, plotindx1], X_pca[:, plotindx2],alpha=0.07)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('LLE Projected data')
plt.show()




