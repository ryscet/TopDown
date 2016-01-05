# -*- coding: utf-8 -*-
"""
This is actually the main script for the analysis using the MNE data structure and type of analysis.
Analysis is done on saved MNE epochs arrays, loaded using LoadEpochs. saved_epochs contains the list of paths
for the windows of interesst that were saved as epochs arrays.
"""

import mne
import numpy as np
from pandas import HDFStore
import matplotlib.pyplot as plt
import glob
#import pandas as pd
import scipy.io as sio
import os
#from mne import io,read_proj, read_selection
#from mne.datasets import sample
#from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
#from mne.time_frequency import induced_power
#from mne.time_frequency import single_trial_power
#import obspy.signal.filter as filters
from scipy.signal import welch as my_welch
from scipy.stats import ttest_ind as t_test
#GLOBALS
eeg_path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Converted_data/signals/';
eeg_names = glob.glob(eeg_path+'*')

events_path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Converted_data/events/';
events_names = glob.glob(events_path +'*')
#########

badChannels = ['LEar', 'REar', 'Iz', 'A1', 'A2', 'AFz', 'FT9', 'FT10', 'FCz']


#List of epochs types that were already saved(thus there is no need to compute them)
saved_epochs = ['/Users/ryszardcetnarski/Desktop/Nencki/TD/MNE/epochs_cue_2sec-epo.fif',
                '/Users/ryszardcetnarski/Desktop/Nencki/TD/MNE/epochs_target_2sec-epo.fif']

from matplotlib import gridspec

def LoadChannels(subID):
    """Load the data saved in matlab format, extracted before with matlab from eeg struct (the event field) of eeglab
    Chanlocs are actually chan names, but most eeg software can infare the location from standard naming"""
    mat_events = [name for name in events_names if '1_TD_ELECTRODES' in name]
    mat_struct = sio.loadmat(mat_events[subID])
    n_channels = len(mat_struct['chanlocs'][0,:])
    chanlocs = [mat_struct['chanlocs'][0,i][0][0] for i in range(0,n_channels)]
    #First make a list with all indexes
    good_chan_idx = [i for i in range(len(chanlocs))]
    #Determine which ones are bad (i.e. do not appear in both caps, but defined manually by a list)
    bad_chan_idx = [idx for idx, item in enumerate(chanlocs) if item in badChannels]
    chanlocs = [chan for chan in chanlocs if chan not in badChannels]
    #Delete their indexes to be used in select slices later
    for index in sorted(bad_chan_idx , reverse=True):
        del good_chan_idx[index]

    return chanlocs, good_chan_idx



def SelectSlices(event_type = 'att_corr', time_type = 'Target', subId = 0, win_size = 1000, forward_window = 250, _good_channels = None):
    ''' Selects slice from datastructures created by MakeDataAndDict()
        Will return [1,1] np.array if demanded events were not found (for example subject made no errors) '''
#Maybe optimize, takes a lot of time, although not in a loop (executed once for every subject. 150 ms * 50 = ~ 8 sec )
    signal = database[ keys['signal'][subId]]
    events = database[ keys[event_type][subId]]
#TODO maybe instead of a completely empty df, make one with correct column names but empty rows,
#Question is which approach is better
    global WIN_SIZE
    WIN_SIZE = win_size
    global FORWARD_WINDOW
    FORWARD_WINDOW = forward_window

    arr_of_slices = np.zeros([1,1,1])

    if(events.empty == False):
#First dim (0) is the amount of slices/epochs, second is the number of electrodes and third is the n_samples. It makes a cube depth(epochs) * height(channels) * width (n_samples)
        arr_of_slices = np.zeros( [len(events[time_type]), len(_good_channels), win_size + forward_window]).astype('float64')


        for idx, time in enumerate(events[time_type]):
            try:
                #Need to transpose because HDF stores electrodes in columns, but MNE in rows
                arr_of_slices[idx,:,:] = signal.iloc[int(time - win_size) : int(time) + forward_window, _good_channels].transpose()
            except:
                pass
    return arr_of_slices

def CollectAllSubjectEpochs():
    """Gathers all the individual 'cubes', trials * electrodes * n_samples and appends them in a massive cube containing all trials, without differentiating per subject.
    """
    global FREQ
    FREQ = 500
    # Initialize an info structure
    global info

    data = []
    events = []
    #All channels are the same after selecting only the common electrodes for the two eeg helmets
    channels = LoadChannels(0)[0]
    montage = 'standard_1005'

    info = mne.create_info(ch_names=channels, ch_types=['eeg' for i in range(0, len(channels))], sfreq=FREQ, montage = montage)

    for i in range(0,46):
        print(i)
        #d is sensor data, e are events
        d, e = CreateEpochs(subID = i)
        data.append(d)
        events.append(e)

    data = np.vstack((data))
    events = np.vstack((events))

    #Create a column of 1,2,3... of the length of the concatenated array, i.e. all events
    event_idx = np.reshape(np.arange(len(events)).T, (len(events), 1))
    events = np.hstack((event_idx, events))
    # t min is a time from the central event around which an epoch is constructed, in our case the presentation of the stimuli. The epoch goes from window size before the stimulus, and forward window after stimulus.
    tmin = -1* (WIN_SIZE / FREQ)


    custom_epochs = mne.EpochsArray(data, info, events, tmin, SANITY_DICT ,baseline=(None, 0))
    #custom_epochs.save('/Users/ryszardcetnarski/Desktop/Nencki/TD/MNE/epochs_cue_2sec-epo.fif')
    return custom_epochs



def CreateEpochs(subID = 0):
    """Creates the MNE epochs array data structure from events and signal"""
    #Process events and use them to cut slices from signal and arrange them in a cube. then also create an array of the length of the depth of the cube (n_trials, i.e. epochs in MNE terminology), which describe the  condition of each trial, attention, motor, correct, etc
    global SANITY_DICT

    sanityDict = {
    'mot_miss' : 0,
    'att_corr' : 1,
    'mot_corr' : 2,
    'att_miss' : 3
    }
    SANITY_DICT = sanityDict

    channel_names, good_indexes = LoadChannels(subID)

    event_id = {}
    allEpochs = {}
    allEvents = []
    #Transpose because HDF stores data in rows but MNE in columns

    for key in keys.keys():
        if key != 'signal':
            type_epochs = SelectSlices(event_type = key, time_type = 'Cue', subId = subID, win_size = 50, forward_window = 1000, _good_channels = good_indexes)
            allEpochs[key] = type_epochs
    diff_epochs_type = []
    for idx, key in enumerate(allEpochs.keys()):
        n_slices = allEpochs[key].shape[0]
        #length 1 is actually an empty array, i.e. no trials of such type, thereofre do not storer anything about it in the events
        if(n_slices > 1):
            diff_epochs_type.append(allEpochs[key])
            duration = list(np.ones(n_slices))
            #Sanity dict makes sure the enumerated keys are always linked to the same integer code
            code = list(np.ones(n_slices)* sanityDict[key])
            events =  np.array([list(elem) for elem in list(zip(duration, code))]).astype('int32')
            allEvents.append(events)
            #event_id must be the same as sanityDict

            event_id[key] = sanityDict[key]
    allEvents = np.vstack(np.array(allEvents))

    events = allEvents

    data = np.concatenate(diff_epochs_type, axis = 0)
    #baseline = (None, 0)  # means from the first instant to t = 0

    return data, events


def PSD(epochs):
    """MNE type of analysis, ununsed so far for anything but visualization. The exact same results are recreated using welch analysis, because I needed all the data points for statistical testing."""
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



def Compute_all_welch(myEpochs, con_1, con_2):
    """The first complete analysis, power spectrum in the determined window using welch method. Two average power spectra are compared using independent t-test"""
#TODO Instead of comparing the power spectra from groups containing all trials in different conditions, intead use the subject averages as the input to the t-test (like in reaction time analysis - same fish measured many times)
    attention = myEpochs[con_1].get_data()
    motor = myEpochs[con_2].get_data()

    all_attention = []
    all_motor = []
    for electrode in range (0,59):

        single_attention = attention[:,electrode,:].reshape(len(attention[:,0,0]),1,len(attention[0,0,:]))
        single_motor = motor[:,electrode,:].reshape(len(motor[:,0,0]),1,len(motor[0,0,:]))

        all_attention.append(Compute_electrode_welch(single_attention))
        all_motor.append(Compute_electrode_welch(single_motor))

        Plot_electrode_welch(all_attention[-1], all_motor[-1], FREQS, electrode, con_1, con_2)

def Compute_electrode_welch(single_electrode):
    all_psd = []
    global FREQS
    for epoch in single_electrode:
        all_psd.append(my_welch(epoch, fs  = 500)[1][0,:])
        freqs = my_welch(epoch, fs  = 500)[0]

    all_psd = np.array(all_psd)
    FREQS = freqs
    return all_psd

def Plot_electrode_welch(attention, motor, freqs, electrode, con1, con2, PLOT_ALL = True):



    mean_att = np.mean(attention, axis = 0)[3:36]
    mean_mot = np.mean(motor, axis = 0)[3:36]

    p_vals = []
    for frequency in range(3,36):
        t,p = t_test(attention[:, frequency], motor[:, frequency])
        p_vals.append(p)

    sigs = [idx for idx,p_val in enumerate(p_vals) if p_val < 0.005]


    if(PLOT_ALL):

        fig = plt.figure()
        fig.suptitle(epochs.info['ch_names'][electrode])
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

        directory = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/All/' + con1 + '_' + con2
        if not os.path.exists(directory):
            os.makedirs(directory)

        savefig('/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/All/' + con1 + '_' + con2 +'/'+ epochs.info['ch_names'][electrode] + '.png')

        if(sigs != []):
            directory = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/Sig/' + con1 + '_' + con2
            if not os.path.exists(directory):
                os.makedirs(directory)
            savefig('/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/Sig/' + con1 + '_' + con2 +'/'+ epochs.info['ch_names'][electrode] + '.png')

            test.set_axis_bgcolor('red')
            print('FOUND DIFFRENCE AT: ')
            print(np.array(sigs) *2)




    if((sigs != []) & (PLOT_ALL == False)):

        fig = plt.figure()
        fig.suptitle(epochs.info['ch_names'][electrode])
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






def OpenDatabase():
    """Load EEG from 64 electrodes x ~30 min at 500 hz (converted (faster) dataset)"""
    hdf_path = "/Users/ryszardcetnarski/Desktop/Nencki/TD/HDF/"
    store = HDFStore(hdf_path +"td_before_database.hdf5")
    return store

def MakeDataAndDict():
    '''Run this first, as it will create globals for other functions
    Database is a all data in HDF format (signals and organized results). Will be loaded into df.
    Keys is a result of dividing all the strings, which are apaths to the HDF database into conditions'''
    global database
    database = OpenDatabase()

    global keys
    keys = database.keys()

    signal_keys = sorted([key for key in keys if 'signal/filtered' in key])

    att_corr_keys = sorted([key for key in keys if 'attention/correct' in key])
    att_miss_keys = sorted([key for key in keys if 'attention/incorrect' in key])

    mot_corr_keys = sorted([key for key in keys if 'motor/correct' in key])
    mot_miss_keys = sorted([key for key in keys if 'motor/incorrect' in key])

    keys = {'signal': signal_keys, 'att_corr' : att_corr_keys, 'mot_corr' : mot_corr_keys,
    'att_miss' : att_miss_keys, 'mot_miss' : mot_miss_keys}

    database, keys

try:
    database
    print('Database loaded')
except:
    print('loading')
    MakeDataAndDict()

def LoadEpochs(idx):
    epochs = mne.read_epochs(saved_epochs[idx], preload= True)
    return epochs

#epochs = CollectAllSubjectEpochs()