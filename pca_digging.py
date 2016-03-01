# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:31:07 2016

@author: ryszardcetnarski
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler


def RunAllPCA():
    bands = ['all_spectrum', 'alpha', 'beta1', 'beta2', 'smr','ratio', 'theta', 'trained', 'ratio']
    for band in bands:
        MakeBlocksArray(band)

def MakeBlocksArray(band):
    path ='/Users/ryszardcetnarski/Desktop/PcaResults/'

    plt.style.use('seaborn-bright')
    db = LoadDatabase()
    all_normed = []
    for name, subject in db.groupby(db.index):
        blocks = ExtractBlocks(subject, 'training', band)
       # return blocks
        #blocks_normed =( zscore(blocks, axis = None).T -  zscore(blocks, axis = None)[:,0][:, np.newaxis].T).T
        blocks_normed = zscore(blocks, axis = None)
        all_normed.append(pd.DataFrame(blocks_normed, index= subject.index))
    all_normed = pd.concat(all_normed)
    all_normed['condition'] = db['condition']

    label_dict = {'plus':0,
                  'minus':1,
                  'control':2,
                  'sham':3}
    color = ['r', 'b','grey','g']




    X = all_normed.ix[:,0:10].values
    y = all_normed.ix[:,10].values
    X_std = StandardScaler().fit_transform(X)

    sklearn_pca = sklearnPCA(n_components=2)
    Y_sklearn = sklearn_pca.fit_transform(X)

    fig = plt.figure()
    fig.suptitle(band, fontweight = 'bold')
    ax = fig.add_subplot(111)
    for idx, row in enumerate(Y_sklearn):
        ax.scatter(row[0], row[1], color = color[label_dict[all_normed.iloc[idx]['condition']]], alpha = 0.5)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend( labels=('plus', 'minus', 'control'))
    legend = ax.get_legend()

    legend.legendHandles[0].set_color('red')
    legend.legendHandles[1].set_color('blue')
    legend.legendHandles[2].set_color('green')

    #fig.savefig(path +'session_avg_pca_'+band+'.png')

  #  return all_normed


def ExtractBlocks(db, train_base, band):
    all_sessions= db[train_base +'_bands']
    all_session_one_band = []
    for session in all_sessions:
        all_session_one_band.append(session.loc[band])
    return np.array(all_session_one_band)



def LoadDatabase():
    df = pd.read_pickle('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/DatabaseTrainBaseUPDATED.pkl')
    return df

