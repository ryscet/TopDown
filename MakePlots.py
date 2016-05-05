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
import matplotlib



def Plot(data, xlabel,fig, offset):
  #  plt.close('all')
    #Beta1 15 -22, #Beta2 22-50
    plt.style.use('seaborn-poster')

    #fig.suptitle(mydict[band] , fontsize = 28)
    i = 1

    path ='/Users/ryszardcetnarski/Desktop/Wykresy/'
    for band, sesje in  sorted(data.items()):
        mydict = {'beta1': 'Beta1 (15-22 Hz)', 'beta2': 'Beta2 (22-45 Hz)', 'alpha': 'Alpha (8-12 Hz)'}
        #ax = fig.add_subplot(111)

        groups_dict = {1:('nMB+', 'red','-'), 2:('CON', 'blue','-'), 3:('MB+', 'red','--')}
        ax = fig.add_subplot(230 + i+ offset)


        i = i+1
        for idx, group in sesje.groupby('grupa', as_index = True):
            x = np.linspace(1, len(group.columns)-1, len(group.columns)-1) #x axis
            mean = group.drop('grupa',1).as_matrix().mean(axis = 0)
            sem = stats.sem(group.drop('grupa',1).as_matrix())

            ax.plot(x,mean, color = groups_dict[idx][1], label = groups_dict[idx][0], linewidth = 3, linestyle = groups_dict[idx][2] )
           # ax.fill_between(np.linspace(1, 8, 8), mean - sem, mean + sem, color = groups_dict[idx][1], alpha = 0.2)
            ax.errorbar(x, mean, yerr=sem, fmt='o', color = groups_dict[idx][1], markeredgecolor = groups_dict[idx][1] )

        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.set_ylabel('')
        ax.xaxis.set_ticks(x)
       # ax.set_xticks()

        ax.set_ylim(-1,2.5)
        ax.set_xlim(0.85,len(group.columns) -1 +0.15)

        for xtick in ax.xaxis.get_major_ticks():
            xtick.label.set_fontsize(22)

        for ytick in ax.yaxis.get_major_ticks():
            ytick.label.set_fontsize(22)


        if(band == 'beta2'):
            ax.yaxis.set_ticks([])



        if(band == 'alpha'):
            ax.set_ylabel('amplitude (z-scored)', fontsize =24)


        if(band == 'beta1'):
            ax.yaxis.set_ticks([])
            ax.set_xlabel(xlabel, fontsize =24)

        if(offset ==0):
            ax.set_title(mydict[band] , fontsize = 28)
            if(band == 'beta2'):
                handles, labels = ax.get_legend_handles_labels()
                legend = ax.legend([handles[2], handles[0], handles[1] ],[labels[2], labels[0], labels[1] ], loc = 'upper right', bbox_to_anchor=(1.35, 1), frameon=False)
                plt.setp(legend.get_title(),fontsize=50)
                plt.setp(plt.gca().get_legend().get_texts(), fontsize='22') #legend 'list' fontsize



        #fig.tight_layout()
    fig.savefig(path + '' +xlabel +'.png')
    fig.savefig(path + '' +xlabel +'.eps')


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

fig = plt.figure(figsize = (45,20))
Plot(LoadBloki(), 'blocks',fig, 0)
Plot(LoadSesje(), 'sessions',fig, 3)

#
#
#def Plot(data, xlabel):
#    plt.close('all')
#    #Beta1 15 -22, #Beta2 22-50
#    plt.style.use('seaborn-poster')
#
#    fig = plt.figure(figsize = (45,8))
#    #fig.suptitle(mydict[band] , fontsize = 28)
#    i = 1
#
#    path ='/Users/ryszardcetnarski/Desktop/Wykresy/'
#    for band, sesje in  sorted(data.items()):
#        mydict = {'beta1': 'Beta1 (15-22 Hz)', 'beta2': 'Beta2 (22-45 Hz)', 'alpha': 'Alpha (8-12 Hz)'}
#        #ax = fig.add_subplot(111)
#
#        groups_dict = {1:('nMB+', 'red','-'), 2:('CON', 'blue','-'), 3:('MB+', 'red','--')}
#        ax = fig.add_subplot(130+i)
#        ax.set_title(mydict[band] , fontsize = 28)
#        i = i+1
#        for idx, group in sesje.groupby('grupa', as_index = True):
#            x = np.linspace(1, len(group.columns)-1, len(group.columns)-1) #x axis
#            mean = group.drop('grupa',1).as_matrix().mean(axis = 0)
#            sem = stats.sem(group.drop('grupa',1).as_matrix())
#
#            ax.plot(x,mean, color = groups_dict[idx][1], label = groups_dict[idx][0], linewidth = 3, linestyle = groups_dict[idx][2] )
#           # ax.fill_between(np.linspace(1, 8, 8), mean - sem, mean + sem, color = groups_dict[idx][1], alpha = 0.2)
#            ax.errorbar(x, mean, yerr=sem, fmt='o', color = groups_dict[idx][1], markeredgecolor = groups_dict[idx][1] )
#
#        ax.spines['top'].set_color('none')
#        ax.spines['right'].set_color('none')
#
#        ax.xaxis.set_ticks_position('bottom')
#        ax.yaxis.set_ticks_position('left')
#
#        ax.set_ylabel('')
#        ax.xaxis.set_ticks(x)
#       # ax.set_xticks()
#
#        ax.set_ylim(-1,2.5)
#        ax.set_xlim(0.85,len(group.columns) -1 +0.15)
#
#        for xtick in ax.xaxis.get_major_ticks():
#            xtick.label.set_fontsize(22)
#
#        for ytick in ax.yaxis.get_major_ticks():
#            ytick.label.set_fontsize(22)
#
#
#        if(band == 'beta2'):
#            handles, labels = ax.get_legend_handles_labels()
#            legend = ax.legend([handles[2], handles[0], handles[1] ],[labels[2], labels[0], labels[1] ], loc = 'upper right', bbox_to_anchor=(1.35, 1), frameon=False)
#            ax.yaxis.set_ticks([])
#           # plt.setp(legend.get_title(),fontsize=50)
#            plt.setp(plt.gca().get_legend().get_texts(), fontsize='22') #legend 'list' fontsize
#
#
#
#        if(band == 'alpha'):
#            ax.set_ylabel('amplitude (z-scored)', fontsize =24)
#
#
#        if(band == 'beta1'):
#            ax.yaxis.set_ticks([])
#            ax.set_xlabel(xlabel, fontsize =24)
#


