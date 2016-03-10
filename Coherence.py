# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:38:48 2016

@author: ryszardcetnarski
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
import pickle
import pandas as pd
from scipy import signal
import mne
import seaborn as sns


def CalcCoherence(x,y):

    fs = 500
    f, Cxy = signal.coherence(x, y, fs, nperseg=256, noverlap = 128, detrend = False)
    plt.semilogy(f, Cxy, alpha = 0.2)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.show()

    # equivalent degrees of freedom: (length(timeseries)/windowhalfwidth)*mean_coherence
    # calculate 95% confidence level
    edof = (len(x)/(256/2)) * Cxy.mean()
    gamma95 = 1.-(0.05)**(1./(edof-1.))
    print(gamma95)
    return np.array(Cxy)


def RunAllSubjects():
    all_subjects = Load_rest_signal()

    long_df = LoopOverDict(all_subjects['before'], 'before')
    long_df.extend(LoopOverDict(all_subjects['after'], 'after'))

    long_df = pd.concat(long_df, ignore_index = True).sort(['frequency','label','subject'])
    fig = plt.figure()
    ax = sns.tsplot(time="frequency", value="coherence",unit="subject", condition="label",
                 data=long_df)

    return long_df

#
def LoopOverDict(mydict, before_after):
    p_coherence = []
    f_coherence = []

    subject_id = 0

    all_df = []
    for name, rest in mydict.items():

        p_coh = CalcCoherence(rest['P3'], rest['P4'])
        p_coherence.append(p_coh)

        f_coh = CalcCoherence(rest['F3'], rest['F4'])
        f_coherence.append(f_coh)

        df_p = MakeLongDf(p_coh, subject_id, 'p_coh_' + before_after)
        all_df.append(df_p)

        df_f =  MakeLongDf(f_coh, subject_id, 'f_coh_'+ before_after)
        all_df.append(df_f)

        subject_id = subject_id+1

    return all_df




def MakeLongDf(coh, subject_id, label):
    df = pd.DataFrame()
    df['frequency'] = np.linspace(0, len(coh)-1, len(coh))
    df['label'] = label
    df['subject'] = [subject_id for i in range(len(coh))]
    df['coherence'] = coh

  #  df = pd.DataFrame([np.array(p_coherence).ravel(), label_list, id_list, freq]).T
   # df.columns =['coherence', 'label', 'subject_id', 'frequency']


    return df#np.array(p_coherence).ravel(),label_list , id_list


     #return rest[0,:], rest[1,:]
        #results = mne.connectivity.spectral_connectivity(rest[np.newaxis,:,:], sfreq = 250)
#    df = pd.DataFrame()
#    df['label'] = ['p_coh_before' for i in range(len(p_coherence))]
#    df['subject_id'] =id_list
#    df['coherence'] = np.array(p_coherence).ravel()
#    return p_coherence, f_coherence, df

#def plot


def Load_rest_signal():


    channels_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/sygnal_rest/electrodes.csv'
    channels = pd.DataFrame()
    channels['electrode'] = pd.read_csv(channels_path).columns
    trainings_electrodes = channels[channels['electrode'].isin( ['F3', 'F4', 'P3', 'P4'])]#.index.values


    path ='/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/sygnal_rest/mat_format/'
    plt.close('all')

  #  alltrainings = shelve.open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Pickles/alltrainings.pickle')
    all_rest = {'before':{}, 'after' :{}}
    #tmp ={}
    full_paths = [x for x in glob.glob(path+'*')]
    #full_paths = [x for x in glob.glob(path+'*') if electrode+'_trening' in x]
    for subject in full_paths:
        #ze name
        short_code = subject[subject.rfind('/')+1:subject.rfind('_') -3]
        print(short_code[10::])
        training = sio.loadmat(subject)['eegToSave']#.swapaxes(0,2).swapaxes(1,2)

       # tmp[short_code[10::]] = training[ trainings_electrodes, :]

        if('_1' in short_code[10::]):
            all_rest['before'][short_code[10:-2]] = PickElectrodes(training, trainings_electrodes)
        else:
            all_rest['after'][short_code[10:-2]] =  PickElectrodes(training, trainings_electrodes)

    return all_rest
    #SavePickle(all_trainings, 'all_trainings')
def PickElectrodes(training, trainings_electrodes):
    electrode_dict = {}

    for idx, row in trainings_electrodes.iterrows():
        electrode_dict[row['electrode']] = training[idx ,:]
    return electrode_dict

