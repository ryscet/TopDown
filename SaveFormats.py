# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:49:02 2016

@author: ryszardcetnarski
"""
from pandas import HDFStore
import scipy.io as sio
import glob
import pickle
import numpy as np
import os


CURRENT_DATABASE ='td_before_database.hdf5'
bef_aft_switch = 'before'

#GLOBALS
eeg_path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Converted_data/signals/';
eeg_names = glob.glob(eeg_path+'*')

events_path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Converted_data/events/';
events_names = glob.glob(events_path +'*')

badChannels = ['LEar', 'REar', 'Iz', 'A1', 'A2', 'AFz', 'FT9', 'FT10', 'FCz']


#define he windows around events here
windows_selected = {'Target_back': 1000, 'Target_forth' :0, 'Cue_back':-100, 'Cue_forth': 1000 }



#Used to create MNE epochs code
sanityDict = {
    'mot_miss' : 0,
    'att_corr' : 1,
    'mot_corr' : 2,
    'att_miss' : 3
    }
#########

def CollectAllNumpy():
    shortName = {'td_before_database.hdf5': 'before', 'td_after_database.hdf5': 'after'}

    all_data = {}
    #Iterate through before and after results
    for _d_base in ['td_before_database.hdf5', 'td_after_database.hdf5']:
        CURRENT_DATABASE = _d_base
        database, keys = MakeDataAndKeys(CURRENT_DATABASE)
        print(_d_base)
        #Iterate thorugh different types of stimuli on screen
        all_stim = {}
        for _w_type in ['Target', 'Cue']:
            print(_w_type)
            #Iterate through all keys, i.e. folders in the HDF database, and select only those that contain timestamps of events (only other folders contain the signal itself)
            gen = (condition for condition in keys.keys() if 'signal' not in condition)
            #iterate through different conditions, like attention correct, motor incorrect ets
            all_conditions = {}
            for condition in gen:
                #Iterate through all subjects
                print(condition)
                all_subs = []
                for subId in range(0,46):
                    print(subId)
                    #Need to load channels for every subject unfortunately, because they have different caps
                    channel_names, good_indexes = LoadChannels(subId)
                    all_subs.append(SelectSlices(condition, _w_type, good_indexes, database[keys['signal'][subId]], database[keys[condition][subId]]))
                # Collect the repsonses for all subjects in a given event in a given condition and save  them to the file
                all_conditions[condition] = all_subs
                path = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/' + shortName[CURRENT_DATABASE] + '/'+ _w_type+ '/'

                if not os.path.exists(path):
                    os.makedirs(path)

                with open(path  + condition +'.p', "wb") as output:
                    pickle.dump(all_subs, output)

            all_stim[_w_type] = all_conditions
        all_data[_d_base] = all_stim
    path_alldata = '/Users/ryszardcetnarski/Desktop/Nencki/TD/Pickle/'
    if not os.path.exists(path_alldata):
        os.makedirs(path_alldata)
    with open(path_alldata + 'AllData.p', "wb") as output:
        pickle.dump(all_data, output)

    return all_data





#--------------FUNCTIONS TO CREATE MNE EPOCHS---------------



#Put here by default values that would raise an error if not overwritten by func passed parameters. Bad way of looking for bugs, change to not-predeifned parameters.
def SelectSlices(event_type, time_type , _good_channels, signal, events):
    """
    It makes a cube depth(epochs) * height(channels) * width (n_samples). Cube contains trials (i.e. epochs) of type defined by event type (att corr itp) and around event defined by time type, for example Target
    Will return [1,1,1] np.array if demanded events were not found (for example subject made no errors)
    """
    win_size = windows_selected[time_type + '_back']
    forward_window = windows_selected[time_type + '_forth']

    global WIN_SIZE
    WIN_SIZE = win_size
    global FORWARD_WINDOW
    FORWARD_WINDOW = forward_window

    arr_of_slices = np.zeros([1,1,1])

#    if(events.empty == False):
#Try because there might have been events codes that never actually occur, like mot miss, or some events were all cut out in eeglab boundaries from a person because of signal noise
    try:
#First dim (0) is the amount of slices/epochs, second is the number of electrodes and third is the n_samples. It makes a cube depth(epochs) * height(channels) * width (n_samples)
        arr_of_slices = np.zeros( [len(events[time_type]), len(_good_channels), win_size + forward_window]).astype('float64')

        for idx, time in enumerate(events[time_type]):
            #Need to transpose because HDF stores electrodes in columns, but MNE in rows
            arr_of_slices[idx,:,:] = signal.iloc[int(time - win_size) : int(time) + forward_window, _good_channels].transpose()
    except:
        print("no events were found for: " + event_type)
        pass

    return arr_of_slices



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

def CollectAllSubjectEpochs(window_type):
    """Gathers all the individual 'cubes', trials * electrodes * n_samples and appends them in a massive cube containing all trials, without differentiating per subject.
        Window type determines from which event the time is selected. It is neccessary to also specify the time back and forth from the central event somwhere downstream (SelectSlices)
        Window types: 'Cue', 'Target', 'Button'
        Toggle comment in second to last line to save
    """
    global subjectLimits
    subjectLimits = []
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
        print(eeg_names[i])
        #d is sensor data, e are events, i is SUBJECT_ID
        d, e = CreateEpochs(i, window_type)
        subjectLimits.append(len(e))
        data.append(d)
        events.append(e)

    data = np.vstack((data))
    events = np.vstack((events))

    #Create a column of 1,2,3... of the length of the concatenated array, i.e. all events
    event_idx = np.reshape(np.arange(len(events)).T, (len(events), 1))
    events = np.hstack((event_idx, events))
    # t min is a time from the central event around which an epoch is constructed, in our case the presentation of the stimuli. The epoch goes from window size before the stimulus, and forward window after stimulus.
    tmin = -1* (WIN_SIZE / FREQ)
#TODO put a try here for the cases where no motor error trials were found
    try:
        custom_epochs = mne.EpochsArray(data, info, events, tmin, SANITY_DICT ,baseline=(None, 0))
    except:
        print('Probably no motor error trials found')

        REDUCED_DICT = {
        'att_corr' : 1,
        'mot_corr' : 2,
        'att_miss' : 3
        }
        custom_epochs = mne.EpochsArray(data, info, events, tmin, REDUCED_DICT ,baseline=(None, 0))

#    custom_epochs.save('/Users/ryszardcetnarski/Desktop/Nencki/TD/MNE/after_epochs_target_2sec-epo.fif')
    return custom_epochs



def CreateEpochs(subID, window_type ):
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
            type_epochs = SelectSlices(event_type = key, time_type = window_type, subId = subID, _good_channels = good_indexes)
           # type_epochs = SelectSlices(event_type = key, time_type = window_type, subId = subID, win_size = 50, forward_window = 1000, _good_channels = good_indexes)
            print(key + ' '+  str(len(type_epochs)))
            allEpochs[key] = type_epochs
    diff_epochs_type = []
    for idx, key in enumerate(allEpochs.keys()):
        n_slices = allEpochs[key].shape[0]
        #length 1 is actually an empty array, i.e. no trials of such type, thereofre do not store anything about it in the events
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





def OpenDatabase(_current_databse):
    """Load EEG from 64 electrodes x ~30 min at 500 hz (converted (faster) dataset)"""
    hdf_path = "/Users/ryszardcetnarski/Desktop/Nencki/TD/HDF/"
    store = HDFStore(hdf_path +_current_databse)
    return store

def MakeDataAndKeys(_current_databse):
    '''Run this first, as it will create globals for other functions
    Database is a all data in HDF format (signals and organized results). Will be loaded into df.
    Keys is a result of dividing all the strings, which are apaths to the HDF database into conditions'''
#Added none because I think sometimes when keys were loaded, loading them again with a new databse was appending instead of substituting, which produced errors downstream
    database = OpenDatabase(_current_databse)

    keys = database.keys()

    signal_keys = sorted([key for key in keys if 'signal/filtered' in key])

    att_corr_keys = sorted([key for key in keys if 'attention/correct' in key])
    att_miss_keys = sorted([key for key in keys if 'attention/incorrect' in key])

    mot_corr_keys = sorted([key for key in keys if 'motor/correct' in key])
    mot_miss_keys = sorted([key for key in keys if 'motor/incorrect' in key])

    keys = {'signal': signal_keys, 'att_corr' : att_corr_keys, 'mot_corr' : mot_corr_keys,
    'att_miss' : att_miss_keys, 'mot_miss' : mot_miss_keys}

    return database, keys

#Select good channels and only those overlapping between the two electrode types


#try:
#    keys
#    print('Database loaded')
#except:
#    print('loading')
#    MakeDataAndDict()