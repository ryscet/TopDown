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


#List of epochs types that were already saved(thus there is no need to compute them)
all_saved_epochs = ['/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/before_Cue.p',
                       '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/before_Target.p',
                       '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/after_Cue.p',
                       '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/after_Target.p']
try:
    ch_names
except:
    ch_names =  pickle.load( open('/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/ch_names.p', "rb" ) )

#TODO plot also erps and confirm a good window is taken - check with and without cutting out event time
#Check why raising error found propably no trial of type'' does not reflect error on the plot, for example a missing plot
#TODO exclude subjects that has less then n correct trials?
#TODO change vertical lines to fill_between
#Todo, add small numbers annotating each line to check if its same eople who produce these high tails in red color
def Run():
    for condition in [['att_corr', 'mot_corr'], ['att_corr', 'att_miss'], ['mot_corr', 'mot_miss'] ]:
        for database_idx in range(0,4):
            CollectAllSubsWelch(condition, database_idx)

def loadNumpy():
    database = []
    for database_idx in range(0,4):
        database.append(pickle.load( open(all_saved_epochs[database_idx], "rb" )))
    return database


def CollectAllSubsWelch(conditions, database_idx):
    all_results = []
    short_db = ['Cue Before', 'Target Before', 'Cue After', 'Target After']

    database = pickle.load( open(all_saved_epochs[database_idx], "rb" ))

    for i in range(0,46):
        print(i)
        subject_results = Compute_all_welch(database[i][conditions[0]], database[i][conditions[1]])
        all_results.append(subject_results)

    Plot_electrode_welch(all_results,short_db[database_idx], conditions)
#    return all_subs


def Compute_all_welch(subject_con_a, subject_con_b):
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
    all_electrodes_a_and_b = {name : [] for name in ch_names}
    for subject in _all_results:
        #a and b refer to conditions, for example att_corr and mot_corr
        for a_electrode, b_electrode, electrode_name in zip(subject['electrodes_a'], subject['electrodes_b'], ch_names):
           all_electrodes_a_and_b[electrode_name].append({condition[0] : a_electrode,  condition[1] : b_electrode})

    for key, electrode in all_electrodes_a_and_b.items():
        fig = plt.figure()
        fig.suptitle(key, fontweight = 'bold')
        a_b_conditions = plt.subplot()
        a_avg, b_avg = [],[]

        for idx, subject_avg in enumerate(electrode):
            a_b_conditions.plot(FREQS, np.log10(subject_avg[condition[0]]), color = 'red', alpha = 0.1)
            a_b_conditions.plot(FREQS, np.log10(subject_avg[condition[1]]), color = 'green', alpha = 0.1)
            a_avg.append(subject_avg[condition[0]])
            b_avg.append(subject_avg[condition[1]])

        a_avg = np.array(a_avg)
        b_avg = np.array(b_avg)

        try:
            a_b_conditions.plot(FREQS, np.log10(a_avg.mean(axis = 0)),label = condition[0], linewidth=2, color = 'green', )
            a_b_conditions.plot(FREQS, np.log10(b_avg.mean(axis = 0)),label = condition[1], linewidth=2, color = 'red')

            a_b_conditions.set_xlim(min(FREQS), max(FREQS))
            a_b_conditions.set_xlabel('Frequency')
            a_b_conditions.set_ylabel('Welch power spectrum')



            p_vals = []
            for frequency in range(0, len(FREQS)):
                t,p = t_test(a_avg[:, frequency], b_avg[:, frequency])
                p_vals.append(p)

            sigs = [idx for idx,p_val in enumerate(p_vals) if p_val < 0.005]

            #save in all
            for sig in sigs:
                #sig times two because the index is half the actuall frequency
                a_b_conditions.vlines(x = sig*2, ymin = -2.5, ymax = 1.5, color = 'blue', linestyle = '--', label = 'alpha 0.005', alpha = 0.2)
             #   test.hlines(y = 0.005, xmin =FREQS[3], xmax = FREQS[36],  color = 'red', linestyle = '--', label = 'alpha 0.005')

            a_b_conditions.legend(loc = 'best')

            path_conditions = condition[0] + '_' + condition[1]
            directory = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/Independent/'+ event_type +'/' + path_conditions + '/All/'
            if not os.path.exists(directory):
                os.makedirs(directory)

            plt.savefig('/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/Independent/'+ event_type +'/' + path_conditions+ '/All/'+ key + '.png')



            #Save also in sigs

            if sigs != []:
                directory_sig = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/Independent/'+ event_type +'/' + path_conditions+ '/Sig/'
                if not os.path.exists(directory_sig):
                    os.makedirs(directory_sig)
                plt.savefig('/Users/ryszardcetnarski/Desktop/Nencki/TD/Figs/Independent/'+ event_type +'/' + path_conditions + '/Sig/'+ key + '.png')
        except:
            print(a_avg)
            print(b_avg)
            print('Probably not enouth trials')
