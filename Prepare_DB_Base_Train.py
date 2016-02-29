# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:55:56 2016

@author: ryszardcetnarski

Reads the trainings and baselines from mat files, from shared folder (originally) and combines them into a
dataframe. This dataframe contains smaller dtaframes that are results of bands per blocks, and are saved
per session (each row of the big df)
"""

import glob
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import csv
import  scipy.stats as stats



def PrepareDatabase():
    info = LoadInfo()
    info['badany'] = info['badany'].str.replace('EID-NFBSS-','')
    trainings = LoadAll_mean_freq('train')

    list_of_df = []
    #Convert string timestamps saved for date and time separately into pandas datetime object
    info['timestamp'] =  pd.to_datetime(info['data'] + ' ' +info['czas'])

    for subject, df in info.groupby('badany'):
    #There are some subjects int the original spreadsheet that we dont have the eeg of :EID-NFBSS-092DF7C4, EID-NFBSS-7EAC5897,EID-NFBSS-9F3106OD,EID-NFBSS-CC2C8039
        if(subject in list(trainings.keys())):
            #Create a list of sessions. Split the cube, where depth is sessions, into individual layers, ie. bands (row) by blocks (columns)
            #Drop ones with nans and zeroes, correct session number is the index from the mat file not the number in the opis (there sessions were incremented even after a failed one)
            valid_sessions = [session for session in trainings[subject] if np.isnan(session).all() == False if np.count_nonzero(session) >0]

            valid_sessions = FillNanBlocks(valid_sessions)

            df['training_bands'] =  [pd.DataFrame(session, index = ['all_spectrum', 'theta', 'alpha', 'smr', 'beta1', 'beta2','beta3', 'trained','ratio']) for session in valid_sessions]
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
    #db = db[['sesja','timestamp','baseline_bands', 'training_bands','days_from_first','days_from_previous','n_blocks', 'bloki', 'original_row_number']]
    return db

def SaveToDisk(df):
    df.to_pickle('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/DatabaseTrainBaseUPDATED.pkl')

def FillNanBlocks(list_of_sessions):
    """With Average. List of sessions already has the all nan sessions removed"""
    for session in list_of_sessions:
        if(np.isnan(session).any()):
            #print(list_of_sessions)
            row_mean = np.nanmean(session,axis=1)
            #print(session)
        #print(row_mean)
           # inds =
            cols_nan = list(set((np.where(np.isnan(session.T))[0])))
            for col in cols_nan:
             #   print(session)
                session[:,col] = row_mean
            #    print(session)
               # print('\n')


            #print(cols_nan)
           # print(session[np.where(np.isnan(session))])# = row_mean)
    #print(list_of_sessions)
    return list_of_sessions



def LoadAll_mean_freq(train_base_rest):
    '''train_base_rest = train | base | rest'''
#TODO check whats up with the overlapping subjects from tura 2 and 3, repeat when normalization amplitude methds are finally decided (divide by sum/mean, take sum/mean)
    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/pasma_' + train_base_rest +'/'

    #Select only file names without proc suffix (procenty) and removing the electrode and Abs_amp prefix
    names = [os.path.basename(x)[11:-4] for x in glob.glob(path+'*') if 'Abs' in x]
    #Need to filter for proc again, to pass it for later subject selection, otherwise names uniques will include procs
    full_paths = [x for x in glob.glob(path+'*') if 'Abs' in x]
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
        #all_subjects[subject] = all_electrodes
    return all_subjects


def AddBaselinesToDb(db):
    """There are less and more baselines then sessions, sometimes baseline is none, sometimes session is missing. Here they are fitted together"""
    db_remainder = db.ix[db['n_blocks'] <11]

    db = db.ix[db['n_blocks'] >=11]
   # db_remainder['baseline_bands'] = [None for i in range(0, len(db_remainder))]
    baselines = LoadAll_mean_freq('base')
    #baselines = RemoveBaselineAtIdx(baselines)
    for key, value in baselines.items():
        mask = ~np.isnan(value).all(axis=1)
        filt = value[mask[:,0],:,:]
        baselines[key] = filt

    all_df=[]
    for subject, df in db.groupby(db.index):
        print(subject)
        print(len(df))
        print(len(baselines[subject]))
        print('\n')
        df['baseline_bands'] = [pd.DataFrame(session, columns = ['amp'], index = ['all_spectrum', 'theta', 'alpha', 'smr', 'beta1', 'beta2','beta3', 'trained','ratio']) for session in baselines[subject]]
        all_df.append(df)

    db = pd.concat(all_df)


    db = pd.concat([db, db_remainder], axis=0).sort('original_row_number')
    return db
#NO LONGER NEEDED AFER USING UPDTATED VALUES FROM KASIE,
#THE ONLY THING TO FIX IS -FEC68D38
#def RemoveBaselineAtIdx(baselines):
#    """This function passes by reference (not copy), so changes to _baseline apply to baseline"""
#    #To delete, at index with Nan's
#    _baselines =baselines
#    to_delete= {
#    '2FE454B7': 4,
#    '888B1F99': 3,
#    'E693F5C0': 6,
#    }
#    for subject,idx in to_delete.items():
##        print(subject)
##        print(_baselines[subject].shape)
##        print(_baselines[subject][idx,:,:])
#        _baselines[subject] =  np.delete(_baselines[subject],idx, axis=0)
#       # print(_baselines[subject].shape)
#
#    return _baselines



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


def PrepareRestingDb():
    """Call this function from other scripts calling this one"""
    names, conditions, electrodes = LoadRestingInfo()
    rest_before = ExtractBands('before','')
    rest_after= ExtractBands('after','')
    rest = pd.concat([rest_before,rest_after], axis =1)
    rest = rest.set_index( [names])

    return rest
    #for idx, name in enumerate(names):
    #    rest_db['name'] = rest[idx,:,:]

   # return rest_db
#def LoadRestingInfo():
#
#    with open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Jacek_RestingState/names.csv') as f:
#        reader = csv.reader(f)
#        names = list(reader)[0]
#
#    with open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Jacek_RestingState/conditions.csv') as f:
#        reader = csv.reader(f)
#        conditions = list(reader)[0]
#
#    with open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Jacek_RestingState/electrodes.csv') as f:
#        reader = csv.reader(f)
#        electrodes = list(reader)[0]
#
#    return names, conditions, electrodes

#def LoadResting(before_after, normed_abs):
#    '''If normed leave normed_abs empty, ("")'''
#    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Jacek_RestingState/'+ normed_abs+before_after+'_fft.mat'
#    mat_struct = sio.loadmat(path)[before_after +'_fft']
#    mat_struct = np.swapaxes(mat_struct, 0,2)
#    mat_struct = np.swapaxes(mat_struct, 1,2)
#    return mat_struct
#
#
#def ExtractBands(before_after, normed_abs):
#    """Normally use this one"""
#    SHIFT = 2
#    bands_dict =  {'theta':(4,8),'alpha': (8,12), 'smr': (12,15), 'beta1':(15,22), 'beta2':(22,30), 'trained':(12,22)}
#    #P3, P4, P5, P6
#    selected_electrodes = [18,20,46,49]
#
#
#    box = LoadResting(before_after, normed_abs)[:,:,selected_electrodes]
#    #In the resulting array, rows are subjects, columns are electrodes
#   # sum_subjec_electrode = np.sum(box, axis = 1)
#  #  alpha = box[:,bands_dict['alpha'][0] - SHIFT :bands_dict['alpha'][1] - SHIFT +1,:]
#
#    band_vals = pd.DataFrame()
#
#    for band_name, f_range in bands_dict.items():
#        # -2 to correct for the shift (starts at 2nd herz) and -1 to correct for diefferences between matlab and python, i.e. starting from 1 and including upperbound
#        #First take the sum along the frequency axis(thus eliminating this axis), then take the average across electrodes
#        #amps =np.mean(np.sum(box[:, f_range[0] - SHIFT -1:f_range[1] - SHIFT ,:], axis =1 ), axis =1)
#        amps =np.mean(np.sum(box[:, f_range[0] - SHIFT -1:f_range[1] - SHIFT ,:], axis =1 ), axis =1)
#        band_vals[band_name + '_' + before_after] =amps
#
#
#    return band_vals


def Load_rest():

    channels_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/channels.csv'
    channels = pd.read_csv(channels_path)
#TODO check whats up with the overlapping subjects from tura 2 and 3, repeat when normalization amplitude methds are finally decided (divide by sum/mean, take sum/mean)
    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/pasma_rest/'
    bands = ['all_spectrum', 'theta', 'alpha', 'smr', 'beta1', 'beta2','beta3', 'trained','ratio']
    #Select only file names without proc suffix (procenty) and removing the electrode and Abs_amp prefix
    names = [os.path.basename(x)[11:-4] for x in glob.glob(path+'*') if 'Abs' in x]
    #Need to filter for proc again, to pass it for later subject selection, otherwise names uniques will include procs
    full_paths = [x for x in glob.glob(path+'*') if 'Abs_amp_OO' in x]
    #Get only unique name of a subject
    unique = list(set(names))
    #iterate over subjects
    all_subjects = {}
    for subject in unique:
        all_electrodes = {}
        #Iterate over electrodes per subject
        #This will iterate through all files unique for a subject, i.e. through electrode recordings F3, F4, P3 P4
        for file in [x for x in full_paths if subject in x]:
            #Swap because the original dimensions(depth *height * width) are block * freq * session, I prefer columns to be blocks and layers to be session, so session * freq * block
            before =  pd.DataFrame(sio.loadmat(file)['freq_amp'].swapaxes(0,2)[0,:,:], index = bands, columns = channels['Channel'])
            after =  pd.DataFrame(sio.loadmat(file)['freq_amp'].swapaxes(0,2)[1,:,:],  index = bands, columns = channels['Channel'])
            all_electrodes['Before'] = before[['P3','P4', 'F3', 'F4']]
            all_electrodes['After'] = after[['P3','P4', 'F3', 'F4']]
        all_subjects[subject.replace('EID-NFBSS-','')] = all_electrodes
        #all_subjects[subject] = all_electrodes
    return all_subjects




#def Z_score(vector):
##Sanity check, should be a numpy array anyways. If not np, then subtraction might not subtract a constant from all elements
#    copy = vector
#    nan_idx = np.isnan(vector)
#    #Compute zscore for non nans
#    vector = vector[~nan_idx]
#    z_score = (vector - np.mean(vector))/np.std(vector)
#    #Substitute non nans with z score and keep the remaining nans
#    copy[~nan_idx] = z_score
#    return copy
#


#def index_containing_substring(the_list, substring):
#    for i, s in enumerate(the_list):
#        if substring in s:
#              return i
#    return None
#def truncate(f, n):
#    '''Truncates/pads a float f to n decimal places without rounding'''
#    s = '{}'.format(f)
#    if 'e' in s or 'E' in s:
#        return '{0:.{1}f}'.format(f, n)
#    i, p, d = s.partition('.')
#    return '.'.join([i, (d+'0'*n)[:n]])