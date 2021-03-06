# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:07:33 2016

@author: ryszardcetnarski
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import scipy.io as sio
import obspy.signal.filter as filters
import glob
import pickle
import pandas as pd
import deepdish as dd
import shelve


#def MakeEnvelope():
#    duration = 1.0
#    fs = 400.0
#    samples = int(fs*duration)
#    t = np.arange(samples) / fs
#
#    signal = chirp(t, 20.0, t[-1], 100.0)
#    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
#
#    analytic_signal = hilbert(signal)
#    amplitude_envelope = np.abs(analytic_signal)
#    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
#    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs
#
#    fig = plt.figure()
#    ax0 = fig.add_subplot(211)
#    ax0.plot(t, signal, label='signal')
#    ax0.plot(t, amplitude_envelope, label='envelope')
#    ax0.set_xlabel("time in seconds")
#    ax0.legend()
#    ax1 = fig.add_subplot(212)
#    ax1.plot(t[1:], instantaneous_frequency)
#    ax1.set_xlabel("time in seconds")
#    ax1.set_ylim(0.0, 120.0)


def SaveTrainings(electrode):
    path ='/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/sygnal_treningi/'
    plt.close('all')

  #  alltrainings = shelve.open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Pickles/alltrainings.pickle')


    full_paths = [x for x in glob.glob(path+'*') if electrode+'_trening' in x]
    all_trainings = {}
    for subject in full_paths:
        print(subject[-12:-4])
        training = sio.loadmat(subject)['trening'].swapaxes(0,2).swapaxes(1,2)
        all_trainings[subject[-12:-4]] = training

    SavePickle(all_trainings, 'all_trainings')
   # return all_trainings
#
def Collect_All_Envelopes():

    min_freq = 12
    max_freq = 22
    all_trainings =LoadPickle('all_trainings')
    i = 1
    all_envelopes ={name : [] for name in all_trainings.keys()}
    for name, trainings in all_trainings.items():
        print(name)
        print(i)
        i = i+1
    #Session, signalIdx, block
        for session in range(0, trainings.shape[0]):
            for block in range(0,trainings.shape[2]):
                single_block = trainings[session,:,block][~np.isnan(trainings[session,:,block])]

                #check if there isnt to many nans
                if(np.count_nonzero(~np.isnan(single_block)) > 1000):
                    #Filter for bands
                    filt = FilterData(single_block, min_freq, max_freq)
                    #Get envelope
                    envelope = filters.envelope(filt)
                    all_envelopes[name].append(envelope)

    SavePickle(all_envelopes, 'all_envelopes')
    return all_envelopes



def SaveEnvelopesByCondition(all_envelopes):
    print('loading')
    #all_envelopes = LoadPickle('all_envelopes')
    print('loaded')
    conditions, exclude = LoadConditionsInfo()

    all_envelopes_conditions =  {'plus':[], 'minus':[], 'sham':[]}
    for subject, training in all_envelopes.items():
        pickle_path ='/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Pickles/'

        #Flatten the list of blocks per subject and per all subject too
        if(subject in conditions.index):
            print('envelopes_conditions/'+subject)
            all_envelopes_conditions[conditions.loc[subject]['condition']].extend( np.array([item for sublist in training for item in sublist]))
            #SavePickle(all_envelopes_conditions, 'envelopes_conditions/'+subject)


        else:
            print('subject not found ' + subject)
    print(1)
    np.save(pickle_path+'envelopes_conditions/'+'plus.npy',  np.array(all_envelopes_conditions['plus']))
    print(2)
    np.save(pickle_path+'envelopes_conditions/'+'minus.npy',  np.array(all_envelopes_conditions['minus']))
    print(3)
    np.save(pickle_path+'envelopes_conditions/'+'sham.npy', np.array(all_envelopes_conditions['sham']))


def PlotEnvelopesByCondition():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pickle_path ='/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Pickles/'

    colors = {'plus':'r', 'minus':'b', 'sham':'g'}
    plus = np.load(pickle_path+'envelopes_conditions/'+'plus.npy')
    minus = np.load(pickle_path+'envelopes_conditions/'+'minus.npy')
    sham = np.load(pickle_path+'envelopes_conditions/'+'sham.npy')

    for c_name, condition in zip(['plus', 'minus', 'sham'],[plus, minus, sham]):
        ax.hist(condition, bins = 25, range = (0,25), normed = True, color = colors[c_name], alpha = 0.5, label = c_name)
    ax.set_xlabel('envelope amplitude')
    ax.set_ylabel('normalized count')
    ax.legend(loc = 'best')


def LoadConditionsInfo():
    subjects_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/subjects_conditions.csv'
    exclude_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/exclude.csv'
    info = pd.read_csv(subjects_path, index_col = 'subject')
    exclude = pd.read_csv(exclude_path)['subject'].tolist()
    return info, exclude#
#        envelope = filters.envelope(filt)
#
#      #  analytic_signal = hilbert(filt)
#       # amplitude_envelope = np.abs(analytic_signal)
##
#        fig = plt.figure()
#        ax1 = fig.add_subplot(311)
#        ax2 = fig.add_subplot(312)
#        ax3 = fig.add_subplot(313)
#
#        ax1.plot(single_block)
#        ax2.plot(filt)
#        ax3.plot(envelope, 'g')
##        ax3.set_ylim(-1,10)
#        #print(subject[-12:-4] + '  ' + str(np.count_nonzero(~np.isnan(filt))) +  '  '+str(np.count_nonzero(~np.isnan(single_block))) +'  '+str(envelope[100]))
#        all_envelopes[subject[-12:-4]] = single_block
#       # ax3.hist(envelope)
#    return all_envelopes

def FilterData(channel, _freqmin, _freqmax):
    b_pass = filters.bandpass(channel, freqmin = _freqmin, freqmax = _freqmax, df = 250)
   # b_stop =filters.bandstop(b_pass, freqmin = 49 ,freqmax = 51, df = 500)
    return b_pass

#    #Get only unique name of a subject
#    unique = list(set(names))
#    #iterate over subjects
#    all_subjects = {}
#    for subject in unique:
#        all_electrodes = {}
#        #Iterate over electrodes per subject
#        #This will iterate through all files unique for a subject, i.e. through electrode recordings F3, F4, P3 P4
#        for file in [x for x in full_paths if subject in x]:
#            #Swap because the original dimensions(depth *height * width) are block * freq * session, I prefer columns to be blocks and layers to be session, so session * freq * block
#            before =  pd.DataFrame(sio.loadmat(file)['freq_amp'].swapaxes(0,2)[0,:,:], index = bands, columns = channels['Channel'])
#            after =  pd.DataFrame(sio.loadmat(file)['freq_amp'].swapaxes(0,2)[1,:,:],  index = bands, columns = channels['Channel'])
#            all_electrodes['Before'] = before[['P3','P4', 'F3', 'F4']]
#            all_electrodes['After'] = after[['P3','P4', 'F3', 'F4']]
#        all_subjects[subject.replace('EID-NFBSS-','')] = all_electrodes
#        #all_subjects[subject] = all_electrodes



def SavePickle(var, name):
    pickle_path ='/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Pickles/'
    #Or is it really pickle
    dd.io.save(pickle_path + name+'.h5', var, compression=None)

#    with open(pickle_path + name +'.pickle', 'wb') as handle:
#        pickle.dump(var, handle)



def LoadPickle(name):
    pickle_path ='/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Pickles/'
    var = dd.io.load(pickle_path + name+'.h5')
    return var

def SaveNumpy(var, name):
    pickle_path ='/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Pickles/'
    #Or is it really pickle
    dd.io.save(pickle_path + name+'.h5', var, compression=None)

#    with open(pickle_path + name +'.pickle', 'wb') as handle:
#        pickle.dump(var, handle)



def LoadNumpy(name):
    pickle_path ='/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Pickles/'
    var = dd.io.load(pickle_path + name+'.h5')
    return var



#def SaveShelve(var,name):
#    pickle_path ='/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Pickles/'
#
#    myShelve = shelve.open(pickle_path+'trainings'+'.shelve')
#    myShelve.update(var)
#    myShelve.close()
#
#
#
#def SaveH5PY():
#     h5f = h5py.File('data.h5', 'w')
#     h5f.create_dataset('dataset_1', data=a)
#     h5f.close()