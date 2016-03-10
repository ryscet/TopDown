# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:45:23 2016

@author: ryszardcetnarski
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.io as sio
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats





def PlotSesje():
    plt.close('all')
    #Beta1 15 -22, #Beta2 22-50
    all_sesje = LoadSesje()
    plt.style.use('grayscale')

    for band, sesje in  all_sesje.items():
        mydict = {'beta1': 'Beta (15-22 Hz)', 'beta2': 'Beta (22-45 Hz)', 'alpha': 'Alpha (8-12 Hz)'}
        fig = plt.figure()
        fig.suptitle(mydict[band], fontsize = 18)
        ax = fig.add_subplot(111)

        groups_dict = {1:('NMB+', 'orange'), 2:('CTR', 'blue'), 3:('MB+', 'red')}
        for idx, group in sesje.groupby('grupa', as_index = True):
            mean = group.drop('grupa',1).as_matrix().mean(axis = 0)
            sem = stats.sem(group.drop('grupa',1).as_matrix())
           # print(sem)
            ax.plot(np.linspace(1, 8, 8),mean, color = groups_dict[idx][1], label = groups_dict[idx][0], linewidth = 3)
           # ax.fill_between(np.linspace(1, 8, 8), mean - sem, mean + sem, color = groups_dict[idx][1], alpha = 0.2)
            ax.errorbar(np.linspace(1, 8, 8), mean, yerr=sem, fmt='o', color = groups_dict[idx][1], markeredgecolor = groups_dict[idx][1] )
        ax.set_xlabel('session')
        ax.set_ylabel('amplitude (z scored)')

       # ax.set_xticks()

        ax.set_ylim(-1,2.5)
        ax.set_xlim(0.85,8.15)


        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1],labels[::-1], loc = 'best')



            #sns.tsplot(data=group.as_matrix(), err_style = 95,ax= ax)
            #return group.drop('grupa',1)
        #return pd.melt(sesje, 'grupa')



def LoadSesje():
    session_columns = []
    for band in ['alpha', 'beta1', 'beta2']:
        session_columns.extend(['%s_%i' %(band, i)for i in range(1,9)])
    df = pd.read_csv('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Od_Kasi_ploty/sesje.csv', names = session_columns)
    groups = pd.read_csv('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Od_Kasi_ploty/group.csv')

    sesje = {}
    sesje['alpha'] =  df.filter(regex='alpha')
    sesje['beta1'] =  df.filter(regex='beta1')
    sesje['beta2'] =  df.filter(regex='beta2')

    for band_name, df in sesje.items():
        df['grupa'] =  groups['grupa']
    return sesje



def LoadBloki():
    session_columns = []
    for band in ['alpha', 'beta1', 'beta2']:
        session_columns.extend(['%s_%i' %(band, i)for i in range(1,11)])
    df = pd.read_csv('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Od_Kasi_ploty/bloki.csv', names = session_columns)
    groups = pd.read_csv('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Od_Kasi_ploty/group.csv')

    bloki = {}
    bloki['alpha'] =  df.filter(regex='alpha')
    bloki['beta1'] =  df.filter(regex='beta1')
    bloki['beta2'] =  df.filter(regex='beta2')

    for band_name, df in bloki.items():
        df['grupa'] =  groups['grupa']

    return bloki

