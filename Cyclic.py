# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:55:56 2016

@author: ryszardcetnarski
"""

import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd



def LoadAll_mean_freq():
#FFT's from tura 3 i 2 were not combined, only the mean freqs were

    electrode = '/F3_'
    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Baseline_mean_freqs/'
    files =  [file for file in glob.glob(path +'*') if electrode in file]
    _all = []
    for file in files:
        #squeeze removes the first dimension, because it's actually a 2d array mean_freq x week, but is stored in a 1 x fre x week format. 1 is removed
        _all.append(pd.DataFrame(np.squeeze(sio.loadmat(file)['freq_amp'],0)))#.dropna(axis = 1, how ='all'))
    return  files, _all

def LoadAll_fft():
#FFT's from tura 3 i 2 were not combined, only the mean freqs were
    electrode = '/F3_'
    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Baseline_ffts/'
    files =  [file for file in glob.glob(path +'*') if electrode in file]
    _all = []
    for file in files:
        #squeeze removes the first dimension, because it's actually a 2d array mean_freq x week, but is stored in a 1 x fre x week format. 1 is removed
        _all.append(sio.loadmat(file)['res_all'])
    return files,  _all



def LoadInfo():
    path_tura2 = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/opis_tura_2.csv'
    info_tura2 = pd.read_csv(path_tura2, delimiter = ',', header = 0).fillna('kasia')

    path_tura3 = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/opis_tura_3.csv'
    info_tura3 = pd.read_csv(path_tura3, delimiter = ',', header = 0).fillna('kasia')

    obie_tury = info_tura2.append(info_tura3, ignore_index=True)
    return obie_tury


def PlotSubject():
    names = ['delta', 'theta', 'alpha', 'smr', 'beta1', 'beta2', 'trained']
    mycolors = ['blue', 'magenta', 'green', 'yellow', 'red', 'violet', 'grey']

    axes = []
    path, subjects = LoadAll_mean_freq()

    for name in names:
        fig_tmp = plt.figure()
        fig_tmp.suptitle(name, fontweight = 'bold')

        axes.append(fig_tmp.add_subplot(111))

    for idx, (path, subject) in enumerate( zip(path, subjects)):
        for _idx, name in enumerate(names):
            axes[_idx].plot(subject[_idx,:] - np.nanmean(subject[_idx,:]), color = mycolors[_idx], alpha =0.5)

def CombineDataAndInfo():
    files, freqs = LoadAll_mean_freq()
    info = LoadInfo()
    myReturn = []
    info['band_fft'] = [np.zeros(7) for i in range(len(info))]
    #Groupy automatically sorts groups
    for name, group in info.groupby('badany'):
        print(name)
        file_idx = index_containing_substring(files, name)
        #Try because there are some subjects in the opis that are not in the files
        if(file_idx is not None):
            for idx,column in freqs[file_idx].iteritems():
                if(~column.isnull().all()):
                   #print(column.values)
                   group['band_fft'].iloc[int(idx)] = column.values
        myReturn.append(group)
    myReturn = pd.concat(myReturn, ignore_index = True)

    myReturn['timestamp'] = pd.to_datetime(myReturn['data'] + ' ' +myReturn['czas'])

    return myReturn
        #print(len(group))
      #  try:
      #  for idx, row in group.iterrows():
            #info['band_fft'].loc[idx] = np.array(freqs[file_idx].icol(tmp_idx))
           # tmp_idx = tmp_idx+1
    #    except:
   #         print('ERROR: ' + name)
   # return info
       # group['band_fft']
        #print(name)
        #print()
      #  print(group)




def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return None

       #np.random.randint(10)]
#        for idx, name in enumerate(names):
#            print(idx)
#            axes[idx].plot([1,2],[1,np.random.randint(10)], color = mycolors[idx], alpha = 0.5)
#            start, end = axes[idx].get_xlim()
#            axes[idx].xaxis.set_ticks(np.arange(start, end, 1))
        #axes[idx].legend(loc = 'best')


#
#
#        if(PlotAll):
#            fig = plt.figure()
#            ax = fig.add_subplot(111)
#            fig.suptitle(path)
#
#


        #colors =cm.jet()
       # crange = np.linspace(0,255,len(subject[0,0,:] ))
#        for idx in range(0, len(subject[0,0,:])):
#            ax.plot(np.log10(subject[0,0:50,idx]), color = cm.afmhot(int(crange[idx])))