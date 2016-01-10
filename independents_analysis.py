# -*- coding: utf-8 -*-
"""
Here we analyze before and after seperately.


This is actually the main script for the analysis using the MNE data structure and type of analysis.
Analysis is done on saved MNE epochs arrays, loaded using LoadEpochs. saved_epochs contains the list of paths
for the windows of interesst that were saved as epochs arrays.


Run CollectAllSubjectEpochs() to aggregate and save (if uncommented) MNE epochs array. An argument window type
can be either 'Cue', 'Target' or 'Button'. It is neccessary to also specify the time back and forth from the central event somwhere downstream (SelectSlices)
Save the selected time windows in Readme file in the HDF directory.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import os
from matplotlib import gridspec

#from mne import io,read_proj, read_selection
#from mne.datasets import sample
#from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
#from mne.time_frequency import induced_power
#from mne.time_frequency import single_trial_power
#import obspy.signal.filter as filters
from scipy.signal import welch as my_welch
from scipy.stats import ttest_ind as t_test


#Var used in OpenDatabase and further MakeDataAndDict to create a new epochs aray. Can be either before or after.
#List of epochs types that were already saved(thus there is no need to compute them)
saved_epochs = ['/Users/ryszardcetnarski/Desktop/Nencki/TD/MNE/before_epochs_cue_2sec-epo.fif',
                '/Users/ryszardcetnarski/Desktop/Nencki/TD/MNE/before_epochs_target_2sec-epo.fif',
                '/Users/ryszardcetnarski/Desktop/Nencki/TD/MNE/after_epochs_cue_2sec-epo.fif',
                '/Users/ryszardcetnarski/Desktop/Nencki/TD/MNE/after_epochs_target_2sec-epo.fif']


def RunAllIndependentAnalysis():
    #ALways make sure this corresponds to saved_epochs global, this list is entered manually

    epochs_shorter = [ 'cue_before', 'target_before', 'cue_after', 'target_after']

    all_comparisons = [['att_corr', 'mot_corr'], ['att_corr', 'att_miss'],['mot_corr', 'mot_miss']]


    for idx, epoch_type in enumerate(saved_epochs):
        _epochs = LoadEpochs(idx)
        for comparison in all_comparisons:
            comparison_type = epochs_shorter[idx]
            Compute_all_welch(_epochs, comparison[0], comparison[1], comparison_type)





def Compute_all_welch(myEpochs, con_1, con_2, comparison_type):
    """The first complete analysis, power spectrum in the determined window using welch method. Two average power spectra are compared using independent t-test"""
#TODO Instead of comparing the power spectra from groups containing all trials in different conditions, intead use the subject averages as the input to the t-test (like in reaction time analysis - same fish measured many times)
#HACK, maybe a better way then a global?
    global ch_names
    ch_names =  myEpochs.info['ch_names']
    try:
        attention = myEpochs[con_1].get_data()
        motor = myEpochs[con_2].get_data()

        all_attention = []
        all_motor = []
#Compute welch
        #For each electrode individually
        for electrode in range (0,59):
            #For a single trial, i.e. epoch (depth dimension)
            single_attention = attention[:,electrode,:].reshape(len(attention[:,0,0]),1,len(attention[0,0,:]))
            single_motor = motor[:,electrode,:].reshape(len(motor[:,0,0]),1,len(motor[0,0,:]))

            all_attention.append(Compute_electrode_welch(single_attention))
            all_motor.append(Compute_electrode_welch(single_motor))

            Plot_electrode_welch(all_attention[-1], all_motor[-1], FREQS, electrode, con_1, con_2, comparison_type)
    except:
        print('-----------NO EVENTS OF TYPE FOUND-----------: ' + comparison_type + ' ' + con_1 + ' ' +con_2 )

def Compute_electrode_welch(single_electrode):
    all_psd = []
    global FREQS
    for epoch in single_electrode:
        all_psd.append(my_welch(epoch, fs  = 500)[1][0,:])
        freqs = my_welch(epoch, fs  = 500)[0]

    all_psd = np.array(all_psd)
    FREQS = freqs
    return all_psd

def Plot_electrode_welch(attention, motor, freqs, electrode, con1, con2, comparison_type, PLOT_ALL = True):


    plt.close('all')
    mean_att = np.mean(attention, axis = 0)[3:36]
    mean_mot = np.mean(motor, axis = 0)[3:36]

    p_vals = []
    for frequency in range(3,36):
        t,p = t_test(attention[:, frequency], motor[:, frequency])
        p_vals.append(p)

    sigs = [idx for idx,p_val in enumerate(p_vals) if p_val < 0.005]


    if(PLOT_ALL):

        fig = plt.figure()
        fig.suptitle(ch_names[electrode])
        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
        att_mot = plt.subplot(gs[0])
        test = plt.subplot(gs[1])

        att_mot.plot(FREQS[3:36], np.log10(mean_att) , color = 'green', label = con1)
        att_mot.plot(FREQS[3:36], np.log10(mean_mot), color = 'red', label = con2)

        test.plot(FREQS[3:36], p_vals, color = 'black', linestyle = '--', label = 'p value')
        test.hlines(y = 0.005, xmin =FREQS[3], xmax = FREQS[36],  color = 'red', linestyle = '--', label = 'alpha 0.005')

        test.set_xlim(FREQS[3], FREQS[36])
        att_mot.set_xlim(FREQS[3], FREQS[36])

        test.set_ylabel('Independent t-test')
        att_mot.set_ylabel('Welch')
        test.set_xlabel('Frequency')
        test.legend(loc = 'best')
        att_mot.legend(loc = 'best')

        directory = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/' + comparison_type + '/All/' + con1 + '_' + con2
        if not os.path.exists(directory):
            os.makedirs(directory)

        savefig('/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/' + comparison_type + '/All/' + con1 + '_' + con2 +'/'+ ch_names[electrode] + '.png')

        if(sigs != []):
            directory = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/' + comparison_type + '/Sig/' + con1 + '_' + con2
            if not os.path.exists(directory):
                os.makedirs(directory)
            savefig('/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/' + comparison_type + '/Sig/' + con1 + '_' + con2 +'/'+ ch_names[electrode] + '.png')

            test.set_axis_bgcolor('red')
            print('FOUND DIFFRENCE AT: ')
            print(np.array(sigs) *2)




    if((sigs != []) & (PLOT_ALL == False)):

        fig = plt.figure()
        fig.suptitle(ch_names[electrode])
        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
        att_mot = plt.subplot(gs[0])
        test = plt.subplot(gs[1])

        att_mot.plot(FREQS[3:36], np.log10(mean_att) , color = 'green', label =con1)
        att_mot.plot(FREQS[3:36], np.log10(mean_mot), color = 'red', label = con2)

        test.plot(FREQS[3:36], p_vals, color = 'black', linestyle = '--', label = 'p value')
        test.hlines(y = 0.005, xmin =FREQS[3], xmax = FREQS[36],  color = 'red', linestyle = '--', label = 'alpha 0.005')

        test.set_xlim(FREQS[3], FREQS[36])
        att_mot.set_xlim(FREQS[3], FREQS[36])

        test.set_ylabel('Independent t-test')
        att_mot.set_ylabel('Welch')
        test.set_xlabel('Frequency')
        test.legend(loc = 'best')
        att_mot.legend(loc = 'best')

        test.set_axis_bgcolor('red')
        print('FOUND DIFFRENCE AT: ')
        print(np.array(sigs) *2)



#---------MNE ANALYSIS--------------------


def PSD(epochs):
    """MNE type of analysis, ununsed so far for anything but visualization.
    The exact same results are recreated using welch analysis,
    because I needed all the data points for statistical testing."""
    for i in range(0, 59):
        fig = plt.figure()
        ax_psd = fig.add_subplot(211)
        ax_erp = fig.add_subplot(212)

        epochs['att_corr'].plot_psd(fmin=2, fmax=70, picks = [i], ax = ax_psd, color = 'green')
        epochs['att_miss'].plot_psd(fmin=2, fmax=70, picks = [i], ax = ax_psd, color = 'red')

        epochs['att_corr'].average(picks = [i]).plot(axes = ax_erp)
        epochs['att_miss'].average(picks = [i]).plot(axes = ax_erp)

        fig.suptitle(epochs.info['ch_names'][i])
        ax_psd.set_xlabel('Frequency Hz')
        break

def ERP(epochs):
    evoked_attention = epochs.average(picks = [0])
    #evoked_motor = epochs['mot_corr'].average()

    #evoked_motor.plot()
    evoked_attention.plot()

def Spectr(epochs_data):
    ch_idx = 9
    n_cycles = 2  # number of cycles in Morlet wavelet
    frequencies = np.arange(7, 30, 3)  # frequencies of interest
    Fs = FREQ # sampling in Hz
    times = epochs.times

    power, phase_lock = induced_power(epochs_data['att_corr'], Fs=Fs, frequencies=frequencies, n_cycles=2, n_jobs=1)

    # baseline corrections with ratio
    power /= np.mean(power[:, :, times < 0], axis=2)[:, :, None]

    pl.subplot(1, 1, 1)
    pl.imshow(20 * np.log10(power[ch_idx]), extent=[times[0], times[-1],
              frequencies[0], frequencies[-1]], aspect='auto', origin='lower')
    pl.xlabel('Time (s)')
    pl.ylabel('Frequency (Hz)')
    pl.title('Induced power (%s)' % evoked.ch_names[ch_idx])
    pl.colorbar()

#ax = epochs['att_corr'].average().plot(gfp = True)
#epochs['mot_corr'].average().plot(gfp = True)



def LoadEpochs(idx):
    """Run this before any other analysis as this will create the epochs structure. Run epochs = LoadEpochs(index of saved_epochs [] of interest)"""
    epochs = mne.read_epochs(saved_epochs[idx], preload= True)
    return epochs

#epochs = CollectAllSubjectEpochs()