# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:39:03 2015
Load HDF database and select slices
@author: ryszardcetnarski
"""
#########
import pandas as pd
import numpy as np
from pandas import HDFStore
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import signal
from scipy.signal import butter, lfilter,freqz
from sklearn.preprocessing import normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


###### GLOBALS
_fs = 500 #Sampling rate in Hz
time_step = 1 / _fs  #Sampling rate in ms
low_pass = 2 #Low cut off
high_pass = 70# High cut off
EXCLUDE_THRESHOLD = 3 #Min amount of trials to make frequency analysis
######


def CallErpPlots():
    
    #PlotSlicesPair(('att_corr', 'att_miss'), 'Target', 0, 0, 'Attention hit/miss')
    #PlotSlicesPair(('mot_corr', 'mot_miss'), 'Target', 0, 0, 'Motor hit/miss')
    for subject in range(0,45):
        #print('subject ' + str(subject))
        PlotSlicesPair(('att_corr', 'mot_corr'), 'Target', subject, 0, 'Attention/Motor hit')
        return
        
        
def CallSpecgrams():
    

    for electrode in range(5,6):
        print(electrode)        
        att_all = []
        mot_all = []        
        for subject in range(0,1):            
            att_all.append(CalcSingleSpecgram('att_corr', 'Target', 0, electrode, 'Attention hit', 1000, PLOT_ON = False))
            mot_all.append(CalcSingleSpecgram('mot_corr', 'Target', 0, electrode, 'Motor hit', 1000, PLOT_ON = False))
            
            
        PlotPairSpecgram(AverageSubjects(att_all), AverageSubjects(mot_all), electrode)
    #return att_all

def AverageSubjects(allSubjects):
    for subject in allSubjects:
        allSpectra = np.mean(np.array([subject['spectr'] for subject in allSubjects]), axis = 0)
        allPeriod = np.mean(np.array([subject['period'] for subject in allSubjects]), axis = 0)
        return {'spectr':allSpectra, 'period':allPeriod, 'extent': allSubjects[0]['extent'], 'p_freqs': allSubjects[0]['p_freqs']}
        
        
def PlotPairSpecgram(att, mot, electrode):
    
######Setup figure
    fig = plt.figure() 
    fig.suptitle('electrode: ' +str(electrode)) 
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1,3]) 
    
    #Spectrograms
    ax_att_spec = plt.subplot(gs[0])
    divider_spec = make_axes_locatable(ax_att_spec)
    ax_mot_spec = divider_spec.append_axes("right",size="100%", pad = 0.1, sharey=ax_att_spec)
    ax_att_spec.set_title("Attention")  
    ax_att_spec.set_ylabel("frequency (Hz)")  

    
    #Periodograms
    ax_att_period = plt.subplot(gs[1])
    ax_att_period.set_ylabel("welsch average")  
    divider_period = make_axes_locatable(ax_att_period)
    ax_mot_period = divider_period.append_axes("right",size="100%", pad = 0.1, sharey=ax_att_period)
    ax_mot_spec.set_title("Motor") 
    ax_mot_spec.set_ylabel("frequency (Hz)")  


    #Difference
    ax_diff = plt.subplot(gs[2])
    divider_diff = make_axes_locatable(ax_diff)
    ax_diff_period= divider_diff.append_axes("bottom",size="30%", pad = 0.1)
    ax_diff.set_title("Attention - Motor")    
    ax_diff_period.set_ylabel("difference in average")  

    plt.setp(ax_mot_spec.get_yticklabels(), visible=False)
    plt.setp(ax_mot_period.get_yticklabels(), visible=False)

######End setup
    
######Plot stuff
    maxFreq = len(att['p_freqs'])/3    
    
    _min =np.amin(np.array([att['spectr'], mot['spectr']]))
    _max=np.amax(np.array([att['spectr'], mot['spectr']]))
    
    im = ax_att_spec.imshow(att['spectr'], extent=att['extent'], aspect='auto', vmin = _min, vmax = _max)
    ax_mot_spec.imshow(mot['spectr'], extent=mot['extent'], aspect='auto', vmin = _min, vmax = _max)
    
    ax_att_period.plot(att['p_freqs'][0:maxFreq], att['period'][0:maxFreq], color = 'r')
    ax_mot_period.plot(mot['p_freqs'][0:maxFreq], mot['period'][0:maxFreq], color = 'r')
    
    ax_diff.imshow(att['spectr'] - mot['spectr'],extent = att['extent'], aspect='auto', vmin = _min, vmax = _max)
    
    ax_diff_period.plot(att['p_freqs'],att['period'] - mot['period'], color = 'r')
#######Done 

#Get the position of the stimulus in time
    xmax = ax_att_spec.get_xlim()[1]
    stim_time = WIN_SIZE * (xmax / (WIN_SIZE + forward_window))
    ax_att_spec.axvline(x=stim_time, linewidth=2, color='k', linestyle = '--')
    ax_mot_spec.axvline(x=stim_time, linewidth=2, color='k', linestyle = '--')
    ax_diff.axvline(x=stim_time, linewidth=2, color='k', linestyle = '--')


# Now adding the colorbar  [left, bottom, width, height]
    cbaxes = fig.add_axes([0.925, 0.1, 0.02, 0.8])     
    fig.colorbar(im, cax = cbaxes)    
    return 
def CalcSingleSpecgram(event_type, time_type , subId, electrode_index, title, winSize = 1000, PLOT_ON = False):
    plt.style.use('ggplot')
    allSpectr = []
    all_Pxx_den = []
#Get the slices for frequency analysis
    slices = SelectSlices(event_type, time_type, subId, electrode_index, winSize)    
#Do not proceed if there is too few valid trials   
    if(len(slices) <= EXCLUDE_THRESHOLD):
        return (None, None), (None, None)
    
    if(PLOT_ON): #Create figures
        fig = plt.figure() 
        fig.suptitle(title + '\n' + 'subject: ' + str(subId) + ' electrode: ' + str(electrode_index) + ' n_trials: ' + str(len(slices)), fontweight = 'bold')
        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])


    for row in slices:
        filt = FilterSignal(row)
#Make a spectrogram version with limmited frequencies, to those specified by the "notch" filter        
        data, freqs, bins, im = my_specgram(filt, NFFT=256, Fs=_fs, noverlap=128, minfreq = low_pass , maxfreq = high_pass)

        Pxx_den, p_freqs = TakeNormSpectrum(filt) #FFT periodogram, also normalizes by dividing by sum
        
        if(PLOT_ON):
            ax1.plot(p_freqs, Pxx_den, alpha = 0.2)
#TODO, normalize???
        allSpectr.append(normalize(data, axis = 0, norm = 'l1' ))
        #allSpectr.append(data)
        all_Pxx_den.append(Pxx_den)
        
    allSpectr =  np.mean(np.array(allSpectr), axis = 0)
    all_Pxx_den = np.array(all_Pxx_den).mean(axis = 0) 
    

#Plot Spectrogram

    allSpectr_Log = allSpectr # Log was already taken by the my_specgram function
    
    xmin, xmax =0, np.amax(bins)
    extent = xmin, xmax, freqs[0], freqs[-1]
    if(PLOT_ON):
        ax0.imshow(allSpectr_Log, extent=extent, aspect='auto')
        ax0.set_ylabel('Frequency (Hz)')
        ax0.set_xticklabels( np.linspace(0,2,5)[::-1])
        ax0.set_xlabel('sec from: ' + time_type)
#Plot Periodogram
        ax1.plot(p_freqs,all_Pxx_den, color = 'r')  
        ax1.set_xlim(low_pass,high_pass)
        ax1.set_ylabel('Normed FFT')
        ax1.set_xlabel('Freqency (Hz)')
    #ax1.set_yscale('log')
    #ax2.pcolormesh(allTrials, vmin = 0, vmax = 0.2)
    

    return {'spectr': allSpectr_Log, 'extent':extent, 'period': all_Pxx_den, 'p_freqs':p_freqs}
    
def FilterSignal(sig):
#Filter the signal with a kinda notch (two filters, low and high pass I guess)
    
#Filter out the electrical artefact
    #filt_a = butter_bandpass_filter(sig, 48, 52, _fs, order=3)
    #filt = sig - filt_a 
#Filter out the irrelevant frequencies
    #filt = butter_bandpass_filter(filt, low_pass, high_pass, _fs, order=6)
    filt = butter_bandpass_filter(sig, low_pass, high_pass, _fs, order=6)
    return filt
    
def TakeNormSpectrum(filt):
#Get the power spectrum
    #Pxx_den = np.abs(np.fft.fft(filt, n = 256, d = time_step))**2
#Alternatively with p welsch #p_freqs, Pxx_den = signal.welch(filt, _fs, nperseg = 64, noverlap = 32)
    #p_freqs = np.fft.fftfreq(filt.size, time_step)
    #idx = np.argsort(p_freqs)
#Why this part?
    #p_freqs = p_freqs[idx]
   # Pxx_den = np.array(Pxx_den[idx]) #/ np.array(Pxx_den[idx]).sum()
    
    p_freqs, Pxx_den = signal.welch(filt, _fs, nperseg = 512,nfft = 1024, noverlap = 0)
    return Pxx_den, p_freqs

def PlotSlicesPair(event_type, time_type , subId, electrode_index, title):
    '''Pass a series of tuples, to plot on a single axis two signals and look for a difference
    '''
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1,1) 

    fig.suptitle(title+ '\n' + 'subject: ' + str(subId) + ' electrode: ' + str(electrode_index), fontweight = 'bold')
    
    slices_A = SelectSlices(event_type[0], time_type, subId, electrode_index)
    slices_B = SelectSlices(event_type[1], time_type, subId, electrode_index)

    mean_A, mean_B = slices_A.mean(axis = 0), slices_B.mean(axis = 0)
    
    std_A, std_B = slices_A.std(axis = 0), slices_B.std(axis = 0)
        
    x_A = np.linspace(0, WIN_SIZE / _fs * 1000, len(mean_A))
    x_B = np.linspace(0, WIN_SIZE / _fs * 1000, len(mean_B)) 
    
    for _slice in slices_A: 
            ax.plot(x_A,_slice, color = 'r', alpha = 0.1)    
    for _slice in slices_B: 
            ax.plot(x_B, _slice, color = 'b', alpha = 0.1)
   
    ax.plot(x_A, mean_A, color = 'r', label = event_type[0] + ': ' + str(len(slices_A[:,0])))
    ax.plot(x_B, mean_B, color = 'b', label = event_type[1] + ': ' + str(len(slices_B[:,0])))
    
    ax.fill_between(x_A, mean_A - std_A, mean_A + std_A, alpha = 0.3, color = 'r')
    ax.fill_between(x_B, mean_B - std_B, mean_B + std_B, alpha = 0.3, color = 'b')
    ax.legend(loc = 'best')
    
    if(len(mean_A) != len(mean_B)):
        ax.annotate('NO DATA', xy=(.5, .5), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center', fontsize = 40)  
                
    ax.set_xlabel('time from ' +  time_type)
    ax.set_ylabel('Normalized EEG')
    
    
def TransformData(Pxx, freq):
    new = []
    for point, hz in zip(Pxx, freq):
        new.append(log10((point**2)/hz)**10)
    
    return np.array(new)
        
        
    
def SelectSlices(event_type = 'att_corr', time_type = 'Target', subId = 0, electrode_index = 0, win_size = 500):
    ''' Selects slice from datastructures created by MakeDataAndDict()
        Will return [1,1] np.array if demanded events were not found (for example subject made no errors) '''
#Maybe optimize, takes a lot of time, although not in a loop (executed once for every subject. 150 ms * 50 = ~ 8 sec )    
    signal = database[ keys['signal'][subId]]
    events = database[ keys[event_type][subId]]
    

#TODO maybe instead of a completely empty df, make one with correct column names but empty rows,
#Question is which approach is better
    global WIN_SIZE 
    WIN_SIZE = win_size
    global forward_window 
    forward_window = 250
    if(events.empty == False):
        
        arr_of_slices = np.zeros([len(events[time_type]),win_size + forward_window]).astype('float64')
        

        for idx, time in enumerate(events[time_type]):
            try:
                arr_of_slices[idx, :] = signal.iloc[int(time - win_size) : int(time) + forward_window, int(electrode_index)]
                print(str(int(time - win_size))   +'   ' + str(int(time) + forward_window))

            except:
                pass
#Sanity check here, might not be working because the array will be of that lenght anyways, even if populated only by zeros
        length = len(signal.iloc[int(time - win_size) : int(time), int(electrode_index)]) 
        
        if(length != win_size):
            print('sub: ' + str(subId)  + ' length: ' + str(length) +  ' event: ' +  event_type + ' idx: ' + str(idx) + ' e time: ' + str(time) + ' ' +keys['signal'][subId])

        return arr_of_slices
    else:
        return np.zeros([1,1]).astype('float64')
        
        
        
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
    
    signal_keys = sorted([key for key in keys if 'signal' in key])
    
    att_corr_keys = sorted([key for key in keys if 'attention/correct' in key])
    att_miss_keys = ([key for key in keys if 'attention/incorrect' in key])
    
    mot_corr_keys = sorted([key for key in keys if 'motor/correct' in key])
    mot_miss_keys = sorted([key for key in keys if 'motor/incorrect' in key])   
    
    keys = {'signal': signal_keys, 'att_corr' : att_corr_keys, 'mot_corr' : mot_corr_keys,   
    'att_miss' : att_miss_keys, 'mot_miss' : mot_miss_keys}    
    
    database, keys
    
def CalcTrialsNum():
    n_sub = 46
    corr = []
    miss = []
    for i in range(0,n_sub):
        corr.append([len(database[keys['att_corr'][i]]), len(database[keys['mot_corr'][i]])])
        miss.append([len(database[keys['att_miss'][i]]), len(database[keys['mot_miss'][i]])])
    return np.array(corr), np.array(miss)
try:
    database
    print('Database loaded')
except:
    print('loading')
    MakeDataAndDict()
    
    

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
    
    
    
##################
    
    
    
    
# modified specgram()
def my_specgram(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
             window=mlab.window_hanning, noverlap=128,
             cmap=None, xextent=None, pad_to=None, sides='default',
             scale_by_freq=None, minfreq = None, maxfreq = None, PLOT_ON = False, **kwargs):
    
    """
    call signature::

      specgram(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=128,
               cmap=None, xextent=None, pad_to=None, sides='default',
               scale_by_freq=None, minfreq = None, maxfreq = None, **kwargs)

    Compute a spectrogram of data in *x*.  Data are split into
    *NFFT* length segments and the PSD of each section is
    computed.  The windowing function *window* is applied to each
    segment, and the amount of overlap of each segment is
    specified with *noverlap*.

    %(PSD)s

      *Fc*: integer
        The center frequency of *x* (defaults to 0), which offsets
        the y extents of the plot to reflect the frequency range used
        when a signal is acquired and then filtered and downsampled to
        baseband.

      *cmap*:
        A :class:`matplotlib.cm.Colormap` instance; if *None* use
        default determined by rc

      *xextent*:
        The image extent along the x-axis. xextent = (xmin,xmax)
        The default is (0,max(bins)), where bins is the return
        value from :func:`mlab.specgram`

      *minfreq, maxfreq*
        Limits y-axis. Both required

      *kwargs*:

        Additional kwargs are passed on to imshow which makes the
        specgram image

      Return value is (*Pxx*, *freqs*, *bins*, *im*):

      - *bins* are the time points the spectrogram is calculated over
      - *freqs* is an array of frequencies
      - *Pxx* is a len(times) x len(freqs) array of power
      - *im* is a :class:`matplotlib.image.AxesImage` instance

    Note: If *x* is real (i.e. non-complex), only the positive
    spectrum is shown.  If *x* is complex, both positive and
    negative parts of the spectrum are shown.  This can be
    overridden using the *sides* keyword argument.

    **Example:**

    .. plot:: mpl_examples/pylab_examples/specgram_demo.py

    """

    #####################################
    # modified  axes.specgram() to limit
    # the frequencies plotted
    #####################################

    # this will fail if there isn't a current axis in the global scope
  
    Pxx, freqs, bins = mlab.specgram(x, NFFT, Fs, detrend,
         window, noverlap, pad_to, sides, scale_by_freq)

        #####################################
    if minfreq is not None and maxfreq is not None:
        Pxx = Pxx[(freqs >= minfreq) & (freqs <= maxfreq)]
        freqs = freqs[(freqs >= minfreq) & (freqs <= maxfreq)]
    #####################################
    
    #Z = 10. * np.log10(Pxx)
    Z = Pxx
    Z = np.flipud(Z)
    

    if(PLOT_ON):
        fig, ax = plt.subplots(1,1) 
        ax = gca()
        # modified here
    
    

        if xextent is None: xextent = 0, np.amax(bins)
        xmin, xmax = xextent
        freqs += Fc
        extent = xmin, xmax, freqs[0], freqs[-1]
        im = ax.imshow(Z, cmap, extent=extent, aspect='auto', **kwargs)

        return Z, freqs, bins, im
    else:
        return Z, freqs, bins, 0