# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:04:40 2016

@author: ryszardcetnarski
"""
import pickle
import numpy as np
from scipy.signal import welch
import os
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr


ALPHA_RANGE = [3,4,5]
SMR_RANGE = [5,6]
BETA1_RANGE = [5,6,7,8,9]
BETA2_RANGE = [10,11,12,13,14]

def CorrelateBandRT():
        events_attention = loadSingleNumpy('Info', 'before', 'Target', 'att_corr')
        fft_attention = loadSingleNumpy('FFT', 'before', 'Target', 'att_corr')
        for electrode in range(0,59):

            fig = plt.figure()
            al = fig.add_subplot(141)
            sm = fig.add_subplot(142)
            b1 = fig.add_subplot(143)
            b2 = fig.add_subplot(144)

            b2.set_xlabel('reaction time')
            al.set_ylabel('alpha')
            sm.set_ylabel('SMR')
            b1.set_ylabel('beta 1')
            b2.set_ylabel('beta 2')

            for event, fft in zip(events_attention, fft_attention):
                event = event.dropna()
                alpha, smr, beta1, beta2 = AvgBand(fft[:,electrode,:])
                event['z_score'] = Z_score(event['RT'])
                stim_present_idx =  np.array(event[event['stim_present'] == 1].index.tolist())
                if(stim_present_idx != []):
                    marker = 'bo'
                    al.plot(event['z_score'].ix[stim_present_idx], alpha[stim_present_idx-1], marker)
                    sm.plot(event['z_score'].ix[stim_present_idx], smr[stim_present_idx -1], marker)
                    b1.plot(event['z_score'].ix[stim_present_idx], beta1[stim_present_idx-1], marker)
                    b2.plot(event['z_score'].ix[stim_present_idx], beta2[stim_present_idx-1], marker)

#                for stim_present, rt, _alpha, _smr, _beta1, _beta2 in zip(event['stim_present'], zscore__rt, alpha, smr, beta1, beta2):
#                    if (stim_present == 1):
#                        al.plot(rt, _alpha, 'bo')
#                        sm.plot(rt, _smr, 'bo')
#                        b1.plot(rt, _beta1, 'bo')
#                        b2.plot(rt, _beta2, 'bo')
            print(electrode)
       #     return


def AvgBand(allTrials):
    '''All trials from a single electrode'''
    #take the mean of column, i.e. individual freqencies that make a given band
    alpha = Z_score(allTrials[:, ALPHA_RANGE].mean(axis = 1))
    smr = Z_score(allTrials[:, SMR_RANGE].mean(axis = 1))
    beta1 =Z_score( allTrials[:, BETA1_RANGE].mean(axis = 1))
    beta2 = Z_score( allTrials[:, BETA2_RANGE].mean(axis = 1))

    return alpha, smr, beta1, beta2


def PlotReactionTimes(condition):
    Normed = True
    results_attention =  loadSingleNumpy('Info', 'before', 'Target', condition)
    Cum = False
    htype = 'bar'
    fig  = plt.figure()
    fig.suptitle('Attention test RT: ' +condition, fontweight = 'bold' )
    z_score_attention = fig.add_subplot(211)
    absolute_attention = fig.add_subplot(212)
    present_zscore = []
    absent_zscore = []

    present_absolute = []
    absent_absolute = []
    for subject in results_attention:
        try:
            subject  = subject.dropna()
            subject['zscore'] = Z_score(subject['RT'])

            present_zscore.extend(subject.loc[subject['stim_present'] == 1]['zscore'])
            absent_zscore.extend(subject.loc[subject['stim_present'] == 0]['zscore'])

            present_absolute.extend(subject.loc[subject['stim_present'] == 1]['RT'])
            absent_absolute.extend(subject.loc[subject['stim_present'] == 0]['RT'])

        except:
            print(subject)

    z_score_attention.hist(np.array(present_zscore), bins = 20, range =(-3,3),cumulative = Cum, histtype = htype, normed = Normed, facecolor='blue', alpha=0.3, label = 'attention present')
    z_score_attention.hist(np.array(absent_zscore), bins = 20, range =(-3,3), cumulative = Cum, histtype = htype, normed = Normed, facecolor='red', alpha=0.3, label = 'attention absent')


    absolute_attention.hist(np.array(present_absolute), bins = 20, range =(0,1000), cumulative = Cum, histtype = htype, normed = Normed, facecolor='blue', alpha=0.3, label = 'attention present')
    absolute_attention.hist(np.array(absent_absolute), bins = 20, range =(0,1000), cumulative = Cum, histtype = htype,  normed = Normed, facecolor='red', alpha=0.3, label = 'attention absent')

    absolute_attention.legend(loc = 'best')
    z_score_attention.legend(loc = 'best')

    absolute_attention.set_ylabel('Absolute')
    z_score_attention.set_ylabel('Z-score')





def Z_score(vector):
#Sanity check, should be a numpy array anyways. If not np, then subtraction might not subtract a constant from all elements
    vector = np.array(vector)
    z_score = (vector - np.mean(vector))/np.std(vector)
    return z_score






def loadAllNumpy(folder):
    """It also saves ffts"""
    database = {}
    path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/'+ folder +'/'
    for training in ['before', 'after']:
        database[training] = {}
        for event in ['Cue', 'Target']:
            database[training][event] = {}
            for condition in ['att_corr', 'mot_corr', 'att_miss', 'mot_miss']:
                database[training][event][condition] = pickle.load( open(path + training +'/' + event +'/' + condition + '.p', "rb" ) )

    return database



def loadSingleNumpy(folder, training, event, condition):
     path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/' + folder + '/' + training + '/' + event + '/' + condition +'.p'
     return pickle.load( open(path, "rb" ) )


def SaveFFT(slices):
    all_fft = []
    for subject in slices:
        fft = np.array(Compute_all_welch_single_condition(subject))
        all_fft.append(swapaxes(fft,0,1))

    return all_fft


def loadAllNumpy_andSaveFFT(folder):
    """It also saves ffts"""
    database = {}
    path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/'+ folder +'/'
    for training in ['before', 'after']:
        database[training] = {}
        for event in ['Cue', 'Target']:
            database[training][event] = {}
            for condition in ['att_corr', 'mot_corr', 'att_miss', 'mot_miss']:
                database[training][event][condition] = pickle.load( open(path + training +'/' + event +'/' + condition + '.p', "rb" ) )
#Only save fft when loading slices, perhaps add a boolean condition not to do it always
                if(folder == 'Slices'):
                    fft = SaveFFT(database[training][event][condition])
                    path_fft = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/'+ 'FFT' +'/'+ training +'/' + event +'/'
                    createDir(path_fft)
                    with open(path_fft + condition +'.p', "wb") as output:
                        pickle.dump(fft, output)
    return database


def Compute_all_welch_single_condition(subject):
    """Returns a list of average fft for all electrodes in a given condition for a single subject"""
    all_electrodes = []
    for electrode in range (0,59):
        #Here we select all trials for a single electrode and reshape the so the depth dimension is n_trial, there is no vertical dimension (it used to be n_electrodes), and horizontal dimension is the signal, i.e. n_samples
        single_electrode = subject[:,electrode,:].reshape(len(subject[:,0,0]),1,len(subject[0,0,:]))
        all_electrodes.append(Compute_electrode_welch(single_electrode))

    return all_electrodes


def Compute_electrode_welch(single_electrode):
    """
    Take the average from a single electrode from all trials. Changed from independent analysis version where
    the average was taken only before plotting in a downstream function Plot_electrode_welch
    """
    all_psd = []
    global FREQS
    for epoch in single_electrode:
        freqs, power = welch(epoch, fs  = 500, nperseg =256, nfft = 256, noverlap = 128)
        #power[0,:], changes a shape so then the np.array easilly converts it so that each row is a single trial fft
        all_psd.append(power[0,:])

    all_psd = np.array(all_psd)
    FREQS = freqs[3:36]
    #return np.mean(all_psd, axis = 0)[3:36]
    return all_psd






def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


