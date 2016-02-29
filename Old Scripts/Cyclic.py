# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:55:56 2016

@author: ryszardcetnarski
"""

import glob
import scipy.io as sio
import numpy as np
import pandas as pd
#from string import letters

import os



def PrepareDatabase():
    info = LoadInfo()
    trainings = LoadAll_mean_freq('Treningi')

    list_of_df = []
    #Convert string timestamps saved for date and time separately into pandas datetime object
    info['timestamp'] =  pd.to_datetime(info['data'] + ' ' +info['czas'])
    #info['delta_from_previous'] = [datetime.datetime.now().date() for i in range(len(info))]
    #info['delta_from_first'] = [datetime.datetime.now().date() for i in range(len(info))]

    for subject, df in info.groupby('badany'):
    #There are some subjects int the original spreadsheet that we dont have the eeg of :EID-NFBSS-092DF7C4, EID-NFBSS-7EAC5897,EID-NFBSS-9F3106OD,EID-NFBSS-CC2C8039
        if(subject in list(trainings.keys())):
            #Create a list of sessions
            #Split the cube, where depth is sessions, into individual layers, ie. bands (row) by blocks (columns)
            #Drop ones with nans and zeroes, correct session number is the index from the mat file not the number in the opis (there sessions were incremented even after a failed one)
            df['training_bands'] =  [pd.DataFrame(session, index = ['delta', 'theta', 'alpha', 'smr', 'beta1', 'beta2', 'trained']) for session in trainings[subject] if np.isnan(session).all() == False if np.count_nonzero(session) >0]
            #Normalize date removes hours, minutes etc, so that 23 houtrs diff is still treated as a whole day
            df['days_from_first'] = (df['timestamp'].apply(pd.datetools.normalize_date) - df['timestamp'].apply(pd.datetools.normalize_date).iloc[0]).dt.days
            df['days_from_previous'] = (df['timestamp'].apply(pd.datetools.normalize_date) - df['timestamp'].apply(pd.datetools.normalize_date).shift()).dt.days

            list_of_df.append(df)
    #Merge groups from groupby
    db = pd.concat(list_of_df, ignore_index =True)

    db.drop(['data', 'czas', 'edf'],inplace=True,axis=1)
    db.set_index('badany', inplace = True)

    #Calculate the number of blocks from info saved in a string
    list_of_ints = [[int(x) for x in y.split()] for y in db['bloki']]
    db['n_blocks'] =[np.sum(x) for x in list_of_ints]
    db['original_row_number'] = np.linspace(0,len(db)-1, len(db))
    db = AddBaselinesToDb(db)
    db = db[['sesja','timestamp','baseline_bands', 'training_bands','days_from_first','days_from_previous','n_blocks', 'bloki', 'original_row_number']]

    return db



def LoadOpis():
    #Make sure not load baseline or other
    opis=pd.read_hdf('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/opis_complete_baseline.hdf5', 'opis')
    return opis



def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return None

def LoadAll_mean_freq(train_base):
    '''train base = Treninigi | Baseline'''
#TODO check whats up with the overlapping subjects from tura 2 and 3, repeat when normalization amplitude methds are finally decided (divide by sum/mean, take sum/mean)
    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/' + train_base +'_mean_freqs/'

    #Select only file names without proc suffix (procenty) and removing the electrode prefix
    names = [os.path.basename(x)[3:-4] for x in glob.glob(path+'*') if 'proc' not in x]
    #Need to filter for proc again, to pass it for later subject selection, otherwise names uniques will include procs
    full_paths = [x for x in glob.glob(path+'*') if 'proc' not in x]
    #Get only unique name of a subject
    unique = list(set(names))
    #iterate over subjects
    all_subjects = {}
    for subject in unique:
        all_electrodes = []
        #Iterate over electrodes per subject
        #This will iterate through all files unique for a subject, i.e. through electrode recordings F3, F4, P3 P4
        for file in [x for x in full_paths if subject in x]:
            #Swap because the original dimensions(depth *height * width) are block * freq * session, I prefer columns to be blocks and layers to be session, so session * freq * block
            all_electrodes.append(sio.loadmat(file)['freq_amp'].swapaxes(0,2))
        all_subjects[subject] = AverageElectrodes(all_electrodes)

    return all_subjects

def ExtractBands(db, train_base, band):
    all_sessions= db[train_base +'_bands']
    all_session_one_band = []
    for session in all_sessions:
        all_session_one_band.append(session['band'])
    return np.array(all_session_one_band)

def AddBaselinesToDb(db):
    """There are less and more baselines then sessions, sometimes baseline is none, sometimes session is missing. Here they are fitted together"""
    db_remainder = db.ix[db['n_blocks'] <11]

    db = db.ix[db['n_blocks'] >=11]
   # db_remainder['baseline_bands'] = [None for i in range(0, len(db_remainder))]
    baselines = LoadAll_mean_freq('Baseline')
    baselines = RemoveBaselineAtIdx(baselines)
    for key, value in baselines.items():
        mask = ~np.isnan(value).all(axis=1)
        filt = value[mask[:,0],:,:]
        baselines[key] = filt

    all_df=[]
    for subject, df in db.groupby(db.index):
        df['baseline_bands'] = [pd.DataFrame(session, columns = ['amp'], index = ['delta', 'theta', 'alpha', 'smr', 'beta1', 'beta2', 'trained']) for session in baselines[subject]]
        all_df.append(df)

    db = pd.concat(all_df)


    db = pd.concat([db, db_remainder], axis=0).sort('original_row_number')
    return db

def RemoveBaselineAtIdx(baselines):
    """This function passes by reference (not copy), so changes to _baseline apply to baseline"""
    #To delete, at index with Nan's
    _baselines =baselines
    to_delete= {
    'EID-NFBSS-2FE454B7': 4,
    'EID-NFBSS-888B1F99': 3,
    'EID-NFBSS-E693F5C0': 6,
    }
    for subject,idx in to_delete.items():
#        print(subject)
#        print(_baselines[subject].shape)
#        print(_baselines[subject][idx,:,:])
        _baselines[subject] =  np.delete(_baselines[subject],idx, axis=0)
       # print(_baselines[subject].shape)

    return _baselines


def LoadInfo():
    path_tura2 = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/opis_tura_2.csv'
    info_tura2 = pd.read_csv(path_tura2, delimiter = ',', header = 0).fillna('kasia')

    path_tura3 = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/opis_tura_3.csv'
    info_tura3 = pd.read_csv(path_tura3, delimiter = ',', header = 0).fillna('kasia')

    obie_tury = info_tura2.append(info_tura3, ignore_index=True)
    obie_tury.drop(['folder mdb', 'raport', 'events', 'examiner'],inplace=True,axis=1)
    #obie_tury.set_index('badany')
    return obie_tury

def PrepareMeanFreqs(cube):
    '''Makes an average from all blocks, so the result are changes per session. Appends columns of Nans for incomplete sessions. Converts 0's to Nan's'''
    averaged = cube.mean(axis = 0)
    if(len(averaged[0,:] < 20)):
        filler = np.empty((7,20 - len(averaged[0,:])))
        filler[:,:] = np.NAN

    fullSize = np.hstack((averaged, filler))
    fullSize[fullSize == 0] = np.NAN
    return fullSize

def AverageElectrodes(all_electrodes):

#Empty array in the shape of the first element
    _sum = np.zeros(all_electrodes[0].shape)
#Sum and divide by n elements
    for electrode in all_electrodes:
        _sum = _sum + electrode
    avg = _sum / len(all_electrodes)

    return avg




def Z_score(vector):
#Sanity check, should be a numpy array anyways. If not np, then subtraction might not subtract a constant from all elements
    copy = vector
    nan_idx = np.isnan(vector)
    #Compute zscore for non nans
    vector = vector[~nan_idx]
    z_score = (vector - np.mean(vector))/np.std(vector)
    #Substitute non nans with z score and keep the remaining nans
    copy[~nan_idx] = z_score
    return copy

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])