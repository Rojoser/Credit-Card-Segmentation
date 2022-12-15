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
import time
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.manifold import locally_linear_embedding
from sklearn.decomposition import PCA
import copy
from scipy.stats import shapiro, anderson


path="/home/jvk/Desktop/propulsion/ML_Group_Challenge/Credit-Card-Segmentation/data/transactions_dropped_150_NaN.csv"
card_df = pd.read_csv(path).fillna(0)
myfeats=card_df.columns.to_list()
print(myfeats)

categorical_features = card_df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_features = card_df.select_dtypes(include=['int', 'float']).columns.tolist()

mynumfeats=list(numeric_features)
std=StandardScaler()
pwtr=PowerTransformer(standardize=False)
qtr=QuantileTransformer(n_quantiles=3)
# pca = PCA(n_components=0.90)
pca = PCA()

transf_df=copy.copy(card_df[numeric_features]).to_numpy()
stdevs=np.std(transf_df,axis=0)
transf_df=transf_df/stdevs
X_pca = pca.fit_transform(transf_df)
# print(transf_df.shape)
# print(X_pca.shape)
# print("")

locquals=[]
locquals2=[]
for varind in range(len(X_pca.T)):
    var=X_pca.T[varind]
    shpro=shapiro(var)
    locquals2.append(shpro.statistic)

plotindx1=locquals2.index(sorted(locquals2)[2])
plotindx2=locquals2.index(sorted(locquals2)[1])
# plotindx2=locquals2.index(sorted(locquals2)[0])
# plotindx1=locquals2.index(sorted(locquals2)[-2])
# plotindx2=locquals2.index(sorted(locquals2)[-1])
plotax1=X_pca[:, plotindx1]
plotax2=X_pca[:, plotindx2]

myind1=np.zeros(len(locquals2))
myind2=np.zeros(len(locquals2))
myind1[plotindx1]=1.
myind2[plotindx2]=1.

myind1UNWRPD=list(pca.inverse_transform(myind1))
print(myind1UNWRPD)
ind1Impactor=myind1UNWRPD.index(max(myind1UNWRPD))
myind1UNWRPD=np.array(myind1UNWRPD)*stdevs
plotax1=plotax1/myind1UNWRPD[ind1Impactor]
# print(plotax1)
# print(ind1Impactor)
myind1UNWRPD=myind1UNWRPD/myind1UNWRPD[ind1Impactor]
# print(myind1UNWRPD)
xstr=numeric_features[ind1Impactor]+" -equivalent"


# print("")
myind2UNWRPD=list(pca.inverse_transform(myind2))
# print(myind2UNWRPD)
ind2Impactor=myind2UNWRPD.index(max(myind2UNWRPD))
if ind2Impactor==ind1Impactor:
    ind2Impactor=myind2UNWRPD.index(sorted(myind2UNWRPD)[-2])

myind2UNWRPD=np.array(myind2UNWRPD)*stdevs
plotax2=plotax2/myind2UNWRPD[ind2Impactor]
# print(plotax2)
# print(ind2Impactor)
myind2UNWRPD=myind2UNWRPD/myind2UNWRPD[ind2Impactor]
# print(myind2UNWRPD)
# print("")
ystr=numeric_features[ind2Impactor]+" -equivalent"







fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(plotax1, plotax2,alpha=0.07)
plt.axis('tight')
# plt.xticks([]), plt.yticks([])
plt.xlabel(xstr)
plt.ylabel(ystr)

plt.title('PCA Projected data')
plt.show()

print("x:", plotindx1)
print("y:", plotindx2)



