# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:42:33 2016

@author: ryszardcetnarski

This script analyses before and after training results, and looks for training effects on attention tests
"""

from scipy.signal import welch as my_welch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import ttest_ind as t_test
import os
import pickle


#List of epochs types that were already saved(thus there is no need to compute them)
before_saved_epochs = ['/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/before_Cue.p',
                       '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/before_Target.p']


after_saved_epochs = ['/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/after_Cue.p',
                       '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/after_Target.p']
try:
    ch_names
except:
    ch_names =  pickle.load( open('/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/ch_names.p', "rb" ) )

#TODO plot also erps and confirm a good window is taken - check with and without cutting out event time
#Check why raising error found propably no trial of type'' does not reflect error on the plot, for example a missing plot
#TODO exclude subjects that has less then n correct trials?
#TODO change vertical lines to fill_between
def Run():
    for condition in ['att_corr', 'mot_corr']:
        for database_idx in [0,1]:
            CollectAllSubsWelch(condition, database_idx)


def CollectAllSubsWelch(condition, database_idx):
    all_results = []
    short_db = ['Cue', 'Target']

    before_database = pickle.load( open(before_saved_epochs[database_idx], "rb" ))
    after_database = pickle.load( open(after_saved_epochs[database_idx], "rb" ))

    for i in range(0,46):
        print(i)
        subject_results = Compute_all_welch(before_database[i][condition], after_database[i][condition])
        all_results.append(subject_results)

    Plot_electrode_welch(all_results,short_db[database_idx], condition)
#    return all_subs


def Compute_all_welch(subject_before, subject_after):
    """The first complete analysis, power spectrum in the determined window using welch method. Two average power spectra are compared using independent t-test"""

    print(subject_before.shape)
    print(subject_after.shape)
    all_before= []
    all_after = []
#Compute welch
    #For each electrode individually
    try:
        for electrode in range (0,59):
            #For a single trial, i.e. epoch (depth dimension)
            single_electrode_before = subject_before[:,electrode,:].reshape(len(subject_before[:,0,0]),1,len(subject_before[0,0,:]))
            single_electrode_after = subject_after[:,electrode,:].reshape(len(subject_after[:,0,0]),1,len(subject_after[0,0,:]))

            all_before.append(Compute_electrode_welch(single_electrode_before))
            all_after.append(Compute_electrode_welch(single_electrode_after))
    except:
        print('!!! Found probaly no valid trial of this type !!!!')
#Uncomment to plot
    return {'electrodes_before': all_before, 'electrodes_after':all_after}
  #  except:
  #      print('-----------NO EVENTS OF TYPE FOUND-----------: ' + comparison_type + ' ' + condition )
def Compute_electrode_welch(single_electrode):
    """
    Take the average from a single electrode from all trials. Changed from independent analysis version where
    the average was taken only before plotting in a downstream function Plot_electrode_welch
    """
    all_psd = []
    global FREQS
    for epoch in single_electrode:
        freqs, power = my_welch(epoch, fs  = 500, nperseg =256, nfft = 256, noverlap = 128)
        #why is it [0,:]?, to change the shape?
        all_psd.append(power[0,:])

    all_psd = np.array(all_psd)
    FREQS = freqs[3:36]
    return np.mean(all_psd, axis = 0)[3:36]



def Plot_electrode_welch(_all_results, event_type, condition):
    plt.close('all')
    #event type, i.e. databse, CUe or Target, conditon, for example 'attention correct

#This loop is to reverse the nesting from subject * electrodes to the other way
    all_electrodes_before_and_after = {name: [] for name in ch_names}
    for subject in _all_results:
        for before_electrode, after_electrode, electrode_name in zip(subject['electrodes_before'], subject['electrodes_after'], ch_names):
           all_electrodes_before_and_after[electrode_name].append({'before':before_electrode,  'after':after_electrode})

    for key, electrode in all_electrodes_before_and_after.items():
        fig = plt.figure()
        fig.suptitle(key, fontweight = 'bold')
        before_after = plt.subplot()
        before_avg, after_avg = [],[]

        for idx, subject_avg in enumerate(electrode):
            before_after.plot(FREQS, np.log10(subject_avg['after']), color = 'red', label = 'after', alpha = 0.1)
            before_after.plot(FREQS, np.log10(subject_avg['before']), color = 'green', label = 'before', alpha = 0.1)
            before_avg.append(subject_avg['before'])
            after_avg.append(subject_avg['after'])

        before_avg = np.array(before_avg)
        after_avg = np.array(after_avg)


        before_after.plot(FREQS, np.log10(before_avg.mean(axis = 0)), linewidth=2, color = 'green')
        before_after.plot(FREQS, np.log10(after_avg.mean(axis = 0)), linewidth=2, color = 'red')

        before_after.set_xlim(min(FREQS), max(FREQS))
        before_after.set_xlabel('Frequency')
        before_after.set_ylabel('Welch power spectrum')


        p_vals = []
        for frequency in range(0, len(FREQS)):
            t,p = t_test(before_avg[:, frequency], after_avg[:, frequency])
            p_vals.append(p)

        sigs = [idx for idx,p_val in enumerate(p_vals) if p_val < 0.005]

        #save in all
        for sig in sigs:
            #sig times two because the index is half the actuall frequency
            before_after.vlines(x = sig*2, ymin = -2.5, ymax = 1.5, color = 'blue', linestyle = '--', label = 'alpha 0.005', alpha = 0.2)
         #   test.hlines(y = 0.005, xmin =FREQS[3], xmax = FREQS[36],  color = 'red', linestyle = '--', label = 'alpha 0.005')



        directory = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/BeforeAfter/'+ event_type +'/' + condition + '/All/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig('/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/BeforeAfter/'+ event_type +'/' + condition + '/All/'+ key + '.png')



        #Save also in sigs

        if sigs != []:
            directory_sig = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/BeforeAfter/'+ event_type +'/' + condition + '/Sig/'
            if not os.path.exists(directory_sig):
                os.makedirs(directory_sig)
            plt.savefig('/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/BeforeAfter/'+ event_type +'/' + condition + '/Sig/'+ key + '.png')

