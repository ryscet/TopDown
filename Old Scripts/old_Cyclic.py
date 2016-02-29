
def Plot_Amp_Per_Session():
    plt.style.use('ggplot')
    files, bands_session = LoadAll_mean_freq_Treningi()
    names = ['delta', 'theta', 'alpha', 'smr', 'beta1', 'beta2', 'trained']
    mycolors = ['blue', 'magenta', 'green', 'yellow', 'red', 'violet', 'grey']

    for (subject, name) in zip (bands_session, files):

        fig  = plt.figure()
        fig.suptitle(name[-12::])
        ax = fig.add_subplot(331)
        n_sessions = np.arange(1,21,1)

        for idx, band_name in enumerate(names):
            fitfunc = lambda p, x: p[0]*cos(2*pi/p[1]*x+p[2]) + p[3]*x # Target function
            errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function

        #For fitting only use the data without nans, cannot use the optimize algorithm otherwise
            time = n_sessions[~np.isnan(subject[idx,:])]
            dataForFitting = Z_score(subject[idx,:])[~np.isnan(subject[idx,:])]

        # Initial guess for the parameters
            p0 = [1.0, 1.0, 1.0, 1.0]
        #find the parameters
            p1, success = optimize.leastsq(errfunc, p0[:], args=(time, dataForFitting ))

        #Plot the real data
            ax = fig.add_subplot(330+idx +1)
        #Plot the function
            ax.plot(n_sessions, Z_score(subject[idx,:]), color = mycolors[idx], marker= 'o')
            ax.plot( time, fitfunc(p1, time), "r--", alpha = 0.3)
            ax.set_ylabel(band_name)

        #Calc rms or oother measure of welness of fit
            rms = sqrt(mean_squared_error(dataForFitting, fitfunc(p1, time)))
            ax.annotate('mean sq err = ' + str(rms)[0:4], xy=(.1, .1), xycoords='axes fraction')
        plt.savefig('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Figury z treningów/' +name[-12:-4] +'.png',dpi=200)
        fig.tight_layout()

def CorrelateFreqs_session():
    sns.set(style="white")

    files, bands_session = LoadAll_mean_freq_Treningi()
    names = ['delta', 'theta', 'alpha', 'smr', 'beta1', 'beta2', 'trained']
    mycolors = ['blue', 'magenta', 'green', 'yellow', 'red', 'violet', 'grey']

    bands_dict = {key: [] for key in names}

    for (subject, name) in zip (bands_session, files):
        for idx, band_name in enumerate(names):
            bands_dict[band_name].extend(Z_score(subject[idx,:])[~np.isnan(subject[idx,:])])

    bands_dataframe = pd.DataFrame.from_dict(bands_dict)
    corr = bands_dataframe.corr(method = 'spearman')

    fig = plt.figure()
    fig.suptitle('Session Pearson Correlations', fontweight = 'bold')

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.set()
    sns.heatmap(corr, mask = mask, annot=True)




    # Generate a mask for the upper triangle

#
#    A = np.ma.array(corr.as_matrix(), mask = mask)
#    fig = plt.figure()
#    fig.suptitle('Session correlations', fontweight = 'bold')
#    ax1 = fig.add_subplot(111)
#    cmap = cm.get_cmap('viridis') # jet doesn't have white color
#   # cmap.set_bad('w') # default value is 'k'
#    tmp = ax1.imshow(A, interpolation="nearest", cmap=cmap)
#    fig.colorbar(tmp)
#
#    xnames =  corr.columns[:-1]
#    xnames = xnames.insert(0,'')
#    ax1.set_xticklabels(xnames )
#    ynames = list(xnames.insert( 7, 'trained'))
#    ynames[0] = ''
#    ynames[1] = ''
#    ax1.set_yticklabels( ynames)
#    ax1.grid(b = False)
    # Set up the matplotlib figure
  #  f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
 #   cmap = sns.diverging_palette(100, 150, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
 #   sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
  #              square=True, xticklabels=2, yticklabels=2,
  #              linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
#
#    for key_1, value_1 in bands_dict.items():
#        fig = plt.figure()
#        fig.suptitle(key_1)
#        idx = 1
#        for key_2, value_2 in bands_dict.items():
#
#            if(key_1 != key_2):
#
#                ax= fig.add_subplot(320+idx)
#                ax.scatter(value_1, value_2, color = mycolors[idx-1], alpha = 0.3)
#                coeff, pval = spearmanr(value_1, value_2)
#
#                k2_1, p_val_1 = normaltest(value_1)
#                k2_2, p_val_2 = normaltest(value_2)
#                if((p_val_1 < 0.05) or (p_val_2 <0.05)):
#                  #  ax.set_axis_bgcolor('red')
#                    print(p_val_1)
#
#
#                ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
#                ax.annotate('spear= ' 'coeff = %4f' % coeff   + '  p= %.4f' %  pval , xy=(.1, .1), xycoords='axes fraction')
#
#                #ax.set_xlabel(key_1)
#                ax.set_ylabel(key_2)
#                idx = idx+1
#        plt.savefig('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Figury/KorelacjeMiedzyPasmami/' + key_1 +'.png')

   # return cnames



#
#
#        allFFT = np.array(badany['band_fft'].tolist())
#        #Select band or mean here
#        points = allFFT[:,3]
#        #Normalize here
#        points = points - points[0]
#
#        time = linspace(0, 100, len(points))
#        p0 = [1.0, 1.0, 1.0, 1.0] # Initial guess for the parameters
#        p1, success = optimize.leastsq(errfunc, p0[:], args=(time, points ))
#
#        fig = plt.figure()
#       # fig.suptitle()
#        ax = fig.add_subplot(111)
#        fft = points
#
#










def PlotSubject():
    names = ['delta', 'theta', 'alpha', 'smr', 'beta1', 'beta2', 'trained']
    mycolors = ['blue', 'magenta', 'green', 'yellow', 'red', 'violet', 'grey']

    axes = []
    path, subjects = LoadAll_mean_freq()

    for name in names:
        fig_tmp = plt.figure()
        fig_tmp.suptitle(name, fontweight = 'bold')

        axes.append(fig_tmp.add_subplot(111))

    for idx, (path, subject) in enumerate( zip(path, subjects)):
        for _idx, name in enumerate(names):
            axes[_idx].plot(subject[_idx,:] - np.nanmean(subject[_idx,:]), color = mycolors[_idx], alpha =0.5)

def CombineDataAndInfo_ForBaseline():
    '''And create additional columns, like time deltas. Filter nans and convert examiner codes'''
    files, freqs = LoadAll_mean_freq_baseline()
    info = LoadInfo()
    myReturn = []
    info['baseline_fft'] = [[] for i in range(len(info))]
    info['timestamp'] =  [datetime.datetime.now().date() for i in range(len(info))]
    info['delta_from_previous'] = [datetime.datetime.now().date() for i in range(len(info))]
    info['delta_from_first'] = [datetime.datetime.now().date() for i in range(len(info))]
    #Groupy automatically sorts groups
    for name, group in info.groupby('badany'):
        print(name)
        #Checking if nem from opis exists in fft files
        file_idx = index_containing_substring(files, name)
        #Try because there are some subjects in the opis that are not in the files
        if(file_idx is not None):
       #     for idx,column in freqs[file_idx].iteritems():
            for idx,session in enumerate(freqs.T):

                if(~column.isnull().all()):
                   #print(column.values)
         #          group['baseline_fft'].iloc[int(idx)] = column.tolist()
         #          group['baseline_fft'].iloc[int(idx)] = column.tolist()
                   group['timestamp'] = pd.to_datetime(group['data'] + ' ' + group['czas'])
                   group['days_from_previous'] = (group['timestamp'] - group['timestamp'].shift()).dt.days
                   group['days_from_first'] = (group['timestamp'] - group['timestamp'].iloc[0]).dt.days
        myReturn.append(group)
    myReturn = pd.concat(myReturn, ignore_index = True)

    myReturn['timestamp'] = pd.to_datetime(myReturn['data'] + ' ' +myReturn['czas'])

    myReturn['examiner'].loc[myReturn['examiner'].str.contains('Cezary')] = 'Cezary'
    myReturn['examiner'].loc[myReturn['examiner'].str.contains('LS')] = 'ls'
#some weird one entry with no name
    myReturn['examiner'].iloc[520] = 'ls'

    myReturn = myReturn[myReturn['baseline_fft'].str.len() != 0]

    myReturn = myReturn.dropna(subset = ['baseline_fft'] )

    myReturn['avg_fft'] = np.array(myReturn['baselinefft'].tolist()).mean(axis = 1)
    myReturn['fft_diff'] = myReturn['avg_fft'] - myReturn['avg_fft'].shift()
    myReturn.to_hdf('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/opis_complete_baseline.hdf5', 'opis')
    return myReturn

def PlotBaselines_Selected():
   names = [ 'theta', 'alpha', 'smr', 'beta1', 'beta2', 'trained']
   mycolors = ['blue', 'magenta', 'green', 'yellow', 'red', 'violet', 'grey']
   files,mean_freq = LoadAll_mean_freq_baseline()
   for name, subject in zip(files, mean_freq):
       fig = plt.figure()
       fig.suptitle(name[-12:-4])
       ax = fig.add_subplot(111)
       idx = 0
       for row in subject[0,1::,:]:
          # print(column)
           ax.plot(row, color = mycolors[idx],label =names[idx] )
           idx = idx+1

       ax.legend()



def GroupbyTrainers():
    complete = LoadOpis()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle('Average fft per trainer', fontweight = 'bold')
    idx = 0
    mycolor = ['r','g','b']
    for name, group in complete.groupby('examiner'):
        bands = np.array(group['band_fft'].tolist())
        #Mean is taken from all frequencies per session
        ax.hist(bands.mean(axis = 1), bins = 20, range = (0.5, 3.5), normed = True, color = mycolor[idx], alpha = 0.5, label = name)
        idx = idx+1
    ax.legend(loc = 'best')
    ax.set_xlabel('average power')
    ax.set_ylabel('normed count per session')

def PlotTimeChanges():
    complete = LoadOpis()

    for name, badany in complete.groupby('badany'):

        fitfunc = lambda p, x: p[0]*cos(2*pi/p[1]*x+p[2]) + p[3]*x # Target function
        errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function

        allFFT = np.array(badany['band_fft'].tolist())
        #Select band or mean here 3- alpha
        points = allFFT[:,3]

        #Normalize here
        #points = points - points[0]

        time = linspace(0, 100, len(points))
        p0 = [1.0, 1.0, 1.0, 1.0] # Initial guess for the parameters
#Uncomment here for fitting

     #   p1, success = optimize.leastsq(errfunc, p0[:], args=(time, points ))

        fig = plt.figure()
       # fig.suptitle()
        ax = fig.add_subplot(111)

        fft = points
#Uncomment here for fitting
       # ax.plot( time, fitfunc(p1, time), "r-")
        #ax.hist(complete['delta_from_previous'], bins = 100)
        ax.plot( badany.index - badany.index[0], fft)
        ax.plot( badany.index- badany.index[0], fft , 'bo', alpha = 0.2)
        ax.set_ylabel('numer sesji')
        ax.set_ylabel('avg fft')
        #ax.set_xlim(0,100)

        path =  '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Figury/Baseliny_z_treningów/'
        plt.savefig(path + name +'.png')
        #ax.set_ylim(-2,2)

def Explore(complete):
    fig = plt.figure()
 #   fig.suptitle('')
    ax = fig.add_subplot(111)
    complete = complete.dropna(subset = ['days_from_previous'])
    ax.plot(complete['days_from_previous'], complete['fft_diff'], 'bo', alpha = 0.2)
  #  ax.set_ylim(0,1.5)
  #  ax.set_xlim(0,40)
    ax.set_xlabel('days from previous training')
    ax.set_ylabel('absolute change in avg fft')


def makeArray(text):
    return np.fromstring(text,sep=' ')



#def LoadAll_fft_baseline():
##FFT's from tura 3 i 2 were not combined, only the mean freqs were
#    electrode = '/F3_'
#    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Baseline_ffts/'
#    files =  [file for file in glob.glob(path +'*') if electrode in file]
#    #Get only unique name of a subject
#    unique = list(set(names))
#    _all = []
#    for file in files:
#        #squeeze removes the first dimension, because it's actually a 2d array mean_freq x week, but is stored in a 1 x fre x week format. 1 is removed
#        _all.append(sio.loadmat(file)['res_all'])
#    return files,  _all







