# -*- coding: utf-8 -*-
"""

Call loadEvents() to get the final processed pandas data structure.
Script to open EEG data and relevant info, previously exported to csv using MATLAB
(pymatbridge too slow, no matlab.engine in currently owned matlab version <2014)

This script creates the events data, first by calling loadEvents then Process events. It will also create
the reaction times.

Actually it was used to save the original .csv (from mat) to HDF,
and also organize the events, into full trials in different conditions.
After HDF was made data load quickly, so no separate openinig script is needed.

"""

#########
import glob
import pandas as pd
import scipy.io as sio
import numpy as np
#import csv
#import itertools
#import h5py
from pandas import HDFStore
import obspy.signal.filter as filters

#GLOBALS
eeg_path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Converted_data/signals/';
eeg_names = sorted(glob.glob(eeg_path+'*'))

events_path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Converted_data/events/';
events_names = sorted(glob.glob(events_path +'*'))

#switch between before and after to load respective datasets
bef_aft_switch = 'after'

#IMPORTANT: .mat files for events are saved with the _TD_EVENTS.mat ending, while the signal is ony _TD.mat


bef_aft_dict = {'before_mat':'1_TD_EVENTS.mat',
                'after_mat':'2_TD_EVENTS.mat',
                'before_hdf':'td_before_database.hdf5',
                'after_hdf':'td_after_database.hdf5'
                }
#########

def storeEEGinHDF():
    """Load EEG from 64 electrodes x ~30 min at 500 hz (large dataset)"""
    h_path = "/Users/ryszardcetnarski/Desktop/Nencki/TD/HDF/"


    all_eeg_names= sorted([name for name in eeg_names if bef_aft_dict[bef_aft_switch + '_mat'].replace("_EVENTS", "") in name])
    store = HDFStore(h_path +bef_aft_dict[bef_aft_switch + '_hdf'])

    #Create a HDF database with a single-precision point (float 32)
    cnt = 0
    for recording in all_eeg_names:
        cnt = cnt + 1
        sname = recording.rfind("/") +1

        subId = recording[sname:-4].replace("-", "_")

        sig = pd.DataFrame(sio.loadmat(recording,struct_as_record=True)['eegToSave']).transpose()
        #Modified here to save  a filtered version from: store[subId + "/signal/f"] =  sig.convert_objects())

        store[subId + "/signal/filtered_30/"] =  sig.convert_objects().apply(FilterData, axis = 0)
        print(cnt)
    store.close()

def SAVE_ChangeDictOrder(_processedEvents):
    '''Change the nesting order for the final HDF database - insted of correct/attention, it will go attention/present/correct etc'''


    h_path = "/Users/ryszardcetnarski/Desktop/Nencki/TD/HDF/"
    #Replace the '_EVENTS' because the path n HDF must match exactly, otherwise it was not savivng anything, weirdo
    all_event_names = sorted([name.replace('_EVENTS', '') for name in events_names if bef_aft_dict[bef_aft_switch + '_mat'] in name])

    store = HDFStore(h_path +bef_aft_dict[bef_aft_switch+ '_hdf'])

    for _data, recording in zip(_processedEvents, all_event_names):
        print('I')
        sname = recording.rfind("/") +1
        subId = recording[sname:-4].replace("-", "_")

        store[subId + '/events/attention/correct'] = _data['correct']['attention'].convert_objects()
        store[subId + '/events/motor/correct'] = _data['correct']['motor'].convert_objects()

        store[subId + '/events/attention/incorrect'] = _data['incorrect']['attention'].convert_objects()
        store[subId + '/events/motor/incorrect'] = _data['incorrect']['motor'].convert_objects()

        #print(_data['incorrect']['motor'].convert_objects())



    store.close()



def FilterData(channel):
    b_pass = filters.bandpass(channel, freqmin = 2, freqmax = 30, df = 500)
   # b_stop =filters.bandstop(b_pass, freqmin = 49 ,freqmax = 51, df = 500)
    return b_pass




def MakeDict():
    """Creates a dict to remap the different markers labels (from different experiment iterations) into a unified description"""
    translate = pd.read_excel('/Users/ryszardcetnarski/Desktop/Nencki/TD/Info/Port_codes_RC.xlsx')
    translate_dict_A = dict(zip(translate['EEG_prof_Grabowskiej'].values,translate['Top down'].values))
    translate_dict_B = dict(zip(translate['EEG_prof_Szelag'].values,translate['Top down'].values))
 #Join two dicts
    merged = translate_dict_A.copy()
    merged.update(translate_dict_B)
    return merged

def ProcessEvents(events):
    """Clear events from boundaries, dvde them between conditions and accuracies.
       events need to exist as a global variable"""
    #events = LoadEvents()

    cue_dict = {'Cue Att R': 'Target Att R',
                'Cue Att L': 'Target Att L',
                'Cue Mot R': 'Target Mot R',
                'Cue Mot L': 'Target Mot L',
                }

    acc_dict = {'Target Att R': 'Button 1 R',
                'Target Att L': 'Button 2 L',
                'Target Mot R': 'Button 1 R',
                'Target Mot L': 'Button 2 L',
                }
    processedEvents = []
#Now the shifted columns is moved one index backwards, thus shwing in the same row what happened after the original column
    for _events in events:
#Make a copy of the event type column for a later groupby, irrespective of L or R position of stim
        _events['type_nodir'] = _events['type'].map(lambda x: x[0: x.rfind(" ")] if 'Button' not in x else 'Button')
#Create boolean columns, to find consecutive events based on certain conditions (target after cue, with no boundary in the middle etc)
        _events['completeTrial'] = np.vectorize(mapVals)(_events['type'], _events['type'].shift(-1), cue_dict)
        _events['accurateTrial'] = np.vectorize(mapVals)(_events['type'], _events['type'].shift(-1), acc_dict)
#Apart from finding the complete trials without boundaries also divide them between correct and incorrect trials
        _events['completeAndAccurateTrial'] = _events['completeTrial'] & _events['accurateTrial'].shift(-1)
        _events['completeAndMissTrial'] = False
        idx = np.array(_events[_events['completeAndAccurateTrial'] == True].index.tolist())
        idx = idx +1
        _events['completeAndAccurateTrial'].iloc[idx] = True
        idx = idx +1
        _events['completeAndAccurateTrial'].iloc[idx] = True
#Look for cases where the trial was complete (no boundary between cue - target - response), but the response was incorrect
        for i in range(0, len(_events) -2):
            if((_events['completeTrial'].iloc[i] == True) &
               (_events['accurateTrial'].iloc[i+1] == False) &
               ('Button' in _events['type'].iloc[i+2])):
                   _events['completeAndMissTrial'].iloc[i:i+3] = True
#In type store only present (R), absent (L) info
        _events['stim'] = _events['type'].map(lambda x: 'absent' if x[-1] == 'L' else 'present')
        correct_complete = GroupTrials(_events.ix[_events['completeAndAccurateTrial']==True])
        incorrect_complete = GroupTrials(_events.ix[_events['completeAndMissTrial']==True])
        processedEvents.append({'correct': correct_complete, 'incorrect' : incorrect_complete})

    return processedEvents


#def mySave(_data,DataName, subjectID):
#    h_path = "/Users/ryszardcetnarski/Desktop/Nencki/TD/HDF/"
#
#    store = HDFStore(h_path +"eeg_database2.hdf5")
#
#    store[subjectID+ "/events/" + DataName] =  sig
#
#    hdf.put(subjectID+ "/events/" + DataName, _data.convert_objects(), format='table', data_columns=True)
#    hdf.close()
#






def GroupTrials(df):
    """Split the consecutive rows cue-target-response rows into 3 columns df.pivot() (unused argument 'index = ')
        further split them between the attention and motor conditon
        keep only time, left/right, and event name (cue, target, response) columns"""

        #Create a vector of repeating numbers [1,1,1,2,2,2,3,3,3]. This is used as an index in multiindex pivot to identify rows that belong to the same trial
    if(len(df['type']) % 3 == 0 ):
        threes = [[i]*3 for i in range(0, int(len(df) / 3) )]
        continous = [item for sublist in threes for item in sublist]
    else:
    #%3 Must be 0, otherwise there is something missing
        print("EVENTS MISSING !!!???")
    #Reindex with the previously generated vector of repeating numbers
    df['newIndex'] = continous
    df = df[['latency','type_nodir', 'newIndex', 'stim' ]]
    #Do not specofy values to create multiindex, the only way to represent the latency and left right simoultaneously
    #After Pivot new columns will be created
    df = df.pivot(index = 'newIndex',  columns='type_nodir')
    #Flattent the nested names: from latency/cue to latency cue
    df.columns = [' '.join(col).strip() for col in df.columns.values]

    #Create empty ones in case there were no trials of such type (for example missed motor)
    att = pd.DataFrame()
    mot = pd.DataFrame()
    if 'latency Cue Att' in df:
        att = RenameColumns(df.loc[df['latency Cue Att'].notnull()].dropna(axis=1,how='all'))
    if 'latency Cue Mot' in df:
        mot = RenameColumns(df.loc[df['latency Cue Mot'].notnull()].dropna(axis=1,how='all'))

    att_mot = {'attention': att, 'motor' : mot}

    return att_mot

def RenameColumns(df):
    '''Renames and removes unwanted columns'''
    df = df.ix[:, 0:4]
    #print(df.columns)
    df.columns = ['Button', 'Cue', 'Target', 'Stim_present']
    df = df[['Cue', 'Target', 'Button', 'Stim_present']]
    df['RT'] = df['Button'] - df['Target']

    d = {'present': 1.0, 'absent': 0.0}
    df['Stim_present'] = df['Stim_present'].map(d)
    return df


def mapVals(curr_event, next_event, myDict):
    """used to compare two strings, if they map in a dictionary, basically if row == row+1,
    where the strings are not literally match, that's why the dict.
    The row+1 is done with a df.shift(-1) before this function"""
    return myDict.get(curr_event) == next_event


def LoadEvents():
    """Load the data saved in matlab format, extracted before with matlab from eeg struct (the event field) of eeglab """
    mergedDict = MakeDict()
    mat_events = sorted([name for name in events_names if bef_aft_dict[bef_aft_switch + '_mat'] in name])
    allEvents = []

   # with h5py.File(h_path +"eeg_database.hdf5", 'w') as eeg_database:
    for recording in mat_events:
        mat_struct = sio.loadmat(recording,struct_as_record=True)['event']
    #Unpack the event times
        latency = np.array(mat_struct[0,:]['latency']).astype('float')
    #Unpack the event names
        stype = np.array([name[0][:] for name in mat_struct[0,:]['type']], dtype = object)
        names = ['latency', 'type']
    #Create the pandas dataframe to store the data
        database = pd.DataFrame(data = [latency,stype]).T
        database.columns = names
        allEvents.append(database.replace({"type": mergedDict}))
    #Organize in columns cue, target, button from previous event by row organization
    allEvents = ProcessEvents(allEvents)
    return allEvents



def LoadChannels(subID):
    """Load the data saved in matlab format, extracted before with matlab from eeg struct (the event field) of eeglab """
    mat_events = sorted([name for name in events_names if '1_TD_ELECTRODES' in name])
    mat_struct = sio.loadmat(mat_events[subID])
    n_channels = len(mat_struct['chanlocs'][0,:])
    chanlocs = [mat_struct['chanlocs'][0,i][0][0] for i in range(0,n_channels)]
    return chanlocs

        #[#x[0][0][0]for x in mat_struct]

