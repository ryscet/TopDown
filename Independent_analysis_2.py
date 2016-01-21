# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:32:04 2016

@author: ryszardcetnarski
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:42:33 2016

@author: ryszardcetnarski

This script analyses before and after training results, and looks for training effects on attention tests
"""

from scipy.signal import welch as my_welch
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import gridspec
from scipy.stats import ttest_ind as t_test
import os
import pickle
from sklearn.preprocessing import normalize as norm
#List of epochs types that were already saved(thus there is no need to compute them)
all_saved_epochs = ['/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/Info/before/Cue/',
                       '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/Info/before/Target/',
                       '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/Info/after/Cue/',
                       '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/Info/after/Target/']
try:
    ch_names
except:
    ch_names =  pickle.load( open('/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/ch_names.p', "rb" ) )

valid_trials_threshold = 5
#TODO plot also erps and confirm a good window is taken - check with and without cutting out event time
#Check why raising error found propably no trial of type'' does not reflect error on the plot, for example a missing plot
#TODO exclude subjects that has less then n correct trials?
#TODO change vertical lines to fill_between
#Todo, add small numbers annotating each line to check if its same eople who produce these high tails in red color
def Run():
    for training in ['before', 'after']:
        for event in ['Target', 'Cue']:
            for conditions in [['att_corr', 'mot_corr'], ['att_corr', 'att_miss'], ['mot_corr', 'mot_miss'] ]:
                CompareConditions(training, event, conditions)


def loadAllNumpy():
    database = {}
    path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/'
    for training in ['before', 'after']:
        database[training] = {}
        for event in ['Cue', 'Target']:
            database[training][event] = {}
            for condition in ['att_corr', 'mot_corr', 'att_miss', 'mot_miss']:
                database[training][event][condition] = pickle.load( open(path + training +'/' + event +'/' + condition + '.p', "rb" ) )
    return database

def loadSingleNumpy(training, event, condition):
     path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/' + training + '/' + event + '/' + condition +'.p'
     return pickle.load( open(path, "rb" ) )






def CompareConditions(training, event, conditions):
    results = {}

    for condition in conditions:
        results[condition] = ReverseSubjectElectrodeNesting(CollectAllSubsWelch(training, event, condition))

    Plot_electrode_welch(results, training, event, conditions)

    return results

def ReverseSubjectElectrodeNesting(list_of_subjects):
    dict_of_electrodes = {name : [] for name in ch_names}
    for subject in list_of_subjects:
        for electrode_ffts, electrode_name in zip(subject, ch_names):
#IMPORTANT: Any normalization would be easiest to implement here as 'tis were the rows are enumerated
            dict_of_electrodes[electrode_name].append(electrode_ffts)

    for key, item in dict_of_electrodes.items():

        dict_of_electrodes[key] = np.array(item)
    return dict_of_electrodes

def CollectAllSubsWelch(training, event, condition):
    all_results = []

    path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/' + training + '/' + event + '/'

    results = pickle.load( open(path + condition +'.p', "rb" ))
    #---------Exclude subjectsthat had very few valid trials-------------------------------------
    results = [subject for subject in results if len(subject) >= valid_trials_threshold]

    for subject in results:
        #Collect a list of all electrodes average fft in a given condition * subjects
        all_results.append(Compute_all_welch_single_condition(subject))

    #Plot_electrode_welch(all_results, training, condition, event)
    return all_results

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
        freqs, power = my_welch(epoch, fs  = 500, nperseg =256, nfft = 256, noverlap = 128)
        #power[0,:], changes a shape so then the np.array easilly converts it so that each row is a single trial fft
        all_psd.append(power[0,:])

    all_psd = np.array(all_psd)
    FREQS = freqs[3:36]
    return np.mean(all_psd, axis = 0)[3:36]



def Plot_electrode_welch(all_results, training, event_type, conditions):
    """Takes for an input a dictionary which has two fields, electrodes_a and b. event_type is a string. condition is a list of strings describing what comparison is made"""
    plt.close('all')

    for (name_a, electrode_a), (name_b, electrode_b) in zip(all_results[conditions[0]].items(), all_results[conditions[1]].items()):
        fig = plt.figure()
        fig.suptitle(name_a + '\n'+  conditions[0] + ' vs ' + conditions[1] + '\n' + event_type, fontweight = 'bold')
        a_b_conditions = fig.add_subplot(111)
#Iterate in two seperate loops since they might (and in fact are for sure) of unequal lengths
        for trial_a in electrode_a:
            #just reshaping
            normed = trial_a / max(trial_a)

            a_b_conditions.plot(FREQS, np.log10(normed), color = 'green', alpha = 0.1)

        for trial_b in electrode_b:
            #just reshaping
            normed = trial_b / max(trial_b)
            a_b_conditions.plot(FREQS, np.log10(normed), color = 'red', alpha = 0.1)

        a_b_conditions.set_xlim(min(FREQS), max(FREQS))
        a_b_conditions.set_xlabel('Frequency')
        a_b_conditions.set_ylabel('Welch power spectrum')

        try:
       #     a_b_conditions.plot(FREQS, np.log10(electrode_a.mean(axis = 0)) / max(np.log10(electrode_a.mean(axis = 0))),label = conditions[0], linewidth=2, color = 'green')
         #   a_b_conditions.plot(FREQS, np.log10(electrode_b.mean(axis = 0)) / max(np.log10(electrode_a.mean(axis = 0))) ,label = conditions[1], linewidth=2, color = 'red')



            p_vals = []
            for frequency in range(0, len(FREQS)):
                t,p = t_test(electrode_a[:, frequency], electrode_b[:, frequency])
                p_vals.append(p)

            sigs = [idx for idx,p_val in enumerate(p_vals) if p_val < 0.005]

            #save in all
            for sig in sigs:
                #sig times two because the index is half the actuall frequency
                a_b_conditions.vlines(x = sig*2, ymin = -2.5, ymax = 1.5, color = 'blue', linestyle = '--', alpha = 0.2)
             #   test.hlines(y = 0.005, xmin =FREQS[3], xmax = FREQS[36],  color = 'red', linestyle = '--', label = 'alpha 0.005')

            a_b_conditions.legend(loc = 'best')

            path_conditions = conditions[0] + '_' + conditions[1]
            directory = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/Independent/'+ event_type +'/' + path_conditions + '/'+ training + '/All/'
            if not os.path.exists(directory):
                os.makedirs(directory)

            plt.savefig('/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/Independent/'+ event_type +'/' + path_conditions+ '/'+ training + '/All/'+ name_a + '.png')



            #Save also in sigs

            if sigs != []:
                directory_sig = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/Independent/'+ event_type +'/' + path_conditions+  '/'+ training + '/Sig/'
                if not os.path.exists(directory_sig):
                    os.makedirs(directory_sig)
                plt.savefig('/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/Independent/'+ event_type +'/' + path_conditions + '/'+ training + '/Sig/'+ name_a + '.png')
        except:
           # print(a_avg)
            #print(b_avg)
            print('Probably not enouth trials')


def Compute_all_welch_two_conditions(subject_con_a, subject_con_b):
    """The first complete analysis, power spectrum in the determined window using welch method. Two average power spectra are compared using independent t-test"""

    all_a= []
    all_b = []
#Compute welch
    #For each electrode individually
    try:
        for electrode in range (0,59):
            #For a single trial, i.e. epoch (depth dimension)
            single_electrode_a = subject_con_a[:,electrode,:].reshape(len(subject_con_a[:,0,0]),1,len(subject_con_a[0,0,:]))
            single_electrode_b = subject_con_b[:,electrode,:].reshape(len(subject_con_b[:,0,0]),1,len(subject_con_b[0,0,:]))

            all_a.append(Compute_electrode_welch(single_electrode_a))
            all_b.append(Compute_electrode_welch(single_electrode_b))
    except:
        print('!!! Found probaly no valid trial of this type !!!!')
#Uncomment to plot
    return {'electrodes_a': all_a, 'electrodes_b':all_b}
  #  except:
  #      print('-----------NO EVENTS OF TYPE FOUND-----------: ' + comparison_type + ' ' + condition )

