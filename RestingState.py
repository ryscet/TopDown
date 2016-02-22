# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:24:40 2016


Analyzes resting states, looking for probabilities of obatining same distribution of before after differences randomly (permutation)
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd

from scipy.stats import spearmanr
from scipy.stats.mstats import normaltest
from sklearn.metrics import mean_squared_error
from random import random


#bands_range_dict =  {'theta':(4,7),'alpha': (8,12), 'smr': (12,15), 'beta1':(15,22), 'beta2':(22,50), 'trained':(12,22)}

def PlotPaired(band, normed_abs):
    normed_abs = ''
    path = '/Users/ryszardcetnarski/Desktop/Figs_RS_Changes/'


    joined= pd.concat([ExtractBands_colstack('before', normed_abs), ExtractBands_colstack('after', normed_abs)], axis = 1)
    befores = joined[band +'_before']
    afters = joined[band +'_after']
    diff = afters - befores

    rising = [ 0 if row_before -  row_after <0  else 1 for row_before, row_after in zip (joined[band + '_before'], joined[band +'_after'])]
    color = ['r', 'b']


    fig = plt.figure()
    fig.suptitle(band, fontweight = 'bold')
    ax_box = fig.add_subplot(131)
    ax_hist = fig.add_subplot(132)
    ax_diff = fig.add_subplot(133)

    wzrosty = [x for x in rising if x > 0]
    ax_box.set_title('n spadki = ' + str(len(wzrosty)) + ' n wzrosty = ' + str((44 - len(wzrosty))))

    # plotting the points
    ax_box.scatter(np.zeros(len(befores)) +1, befores, alpha =0.5)
    ax_box.scatter(np.ones(len(afters)) +1, afters, alpha = 0.5)
    ax_box.boxplot([befores, afters])
    ax_box.set_xticklabels(['before', 'after'])


    joined =pd.concat([befores, afters], axis = 0, ignore_index = True).as_matrix()
    _max = max(joined)
    _min = min(joined)
    #Histogram
   # ax_hist.hist(pd.concat([befores, afters], axis = 0, ignore_index = True).as_matrix(), bins = 15, normed = False)
    ax_hist.hist(befores, bins = 15, range = (_min, _max), normed = False, color = 'magenta', alpha = 0.5, label = 'before')
    ax_hist.hist(afters, bins = 15, normed = False,range = (_min, _max), color = 'g', alpha = 0.5, label = 'after' )
    ax_hist.legend(loc ='best')

    ax_diff.hist(diff, bins = 15, normed = False, color = 'b',alpha = 0.5, label = 'after - before' )
    ax_hist.set_xlabel(normed_abs +'amplitude  distribution')
    ax_diff.set_xlabel('differences distribution')
    ax_diff.axvline(0, color = 'b', linestyle = 'dashed', linewidth = 2, label = '0 - środek')

    ax_hist.tick_params(axis='x', labelsize=5)
    ax_diff.tick_params(axis='x', labelsize=5)

    # plotting the lines


    for i in range(len(befores)):
        ax_box.plot( [1,2], [befores[i], afters[i]], c=color[rising[i]], alpha = 0.15)
        #ax_box.plot( [1,2], [rel_befores[i], rel_afters[i]], c=color[rising[i]], alpha = 0.15)
    #path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Figury/RestingState/BeforeAfter_Regression/InsideFreq/'

    fig.savefig(path + band + '_' +normed_abs+'_ditributions.png', dpi =300)
    tmp = {'diff' : diff, 'before': befores}
    fig2 = sns.jointplot(y ='diff', x='before' , data=pd.DataFrame(tmp));
    fig2.set_axis_labels( band+' before', band +' difference (after - before)')
    fig2.savefig(path + band + '_' +normed_abs+'_correlation.png', dpi = 300)
    return tmp
def RunAllBands():
    for band in ['alpha', 'smr', 'beta1', 'beta2', 'theta', 'trained']:

        PlotPaired(band, 'normed_')
        PlotPaired(band, '')
#        print(band)
#        print('independent normed')
#        Permutation_Independent(band, 'normed_')
#        print('paired normed')
#
#        Permutation_Paired(band, 'normed_')
#        print('independent')
#
#        Permutation_Independent(band, '')
#        print('paired')
#
#        Permutation_Paired(band, '')


def Permutation_Paired(band, normed_abs):

    path = '/Users/ryszardcetnarski/Desktop/Permutations/Scatter/Paired/'

    cond_dict = {'plus': 0, 'minus':1, 'sham':2, 'control':3}
    cond_colors = ['red', 'blue', 'grey', 'yellow']
    n_perm = 10#0
    joined= pd.concat([ExtractBands_colstack('before', normed_abs), ExtractBands_colstack('after', normed_abs)], axis = 1)


    names, conditions, electrodes = LoadRestingInfo()


    joined['conditions'] = conditions
    joined['names'] = names

    befores = joined[band+'_before'].as_matrix()
    afters = joined[band+'_after'].as_matrix()


    fig = plt.figure()
    fig.suptitle(band  + '\npaired')
    ax1 = fig.add_subplot(111)

    original_results = CalcDifferencesRatio(befores,afters)
    perm_results = {'coeff': [], 'pval': [], 'sum_ratio':[], 'mean_ratio' : [], 'count_ratio':[], 'avg': []}
    colors = [(1,1,1)] + [(random(),random(),random()) for i in range(255)]
    for i in range(0, n_perm):
        afters_p = np.random.permutation(afters)
        tmp_dict = CalcDifferencesRatio(befores, afters_p)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter( afters_p - befores,befores, color = colors[i], alpha = 0.2, edgecolors='w', s = 100)
        ax.scatter( afters - befores,befores, color = 'r', alpha = 0.5, edgecolors='w', s = 100)

        ax1.scatter( afters_p - befores,befores, color = 'b', alpha = 0.02, edgecolors='w', s = 100, label = 'permutation' if i == 0 else "")

       # ax.set_xlim(-0.4,0.4)

        for key, value in tmp_dict.items():
            perm_results[key].append(value)
    PlotPermutation(original_results, perm_results, band, 'Paired', normed_abs)
 #   return original_results, perm_results
    #ax1.scatter( afters - befores,befores,color = 'r', alpha = 0.8, edgecolors='w', s = 100, label = 'original')

    for i in range(0,len(afters)):
        ax1.scatter( afters[i] - befores[i],befores[i],color = cond_colors[cond_dict[joined['conditions'].iloc[i]]], alpha = 0.8, edgecolors='w', s = 100)

    ax1.set_xlabel('After - Before')
    ax1.set_ylabel('Before')
    ax1.legend(loc = 'best')
    fig.savefig(path + band+ '_'+ normed_abs+ '.png', dpi =300)

def Permutation_Independent(band, normed_abs):
    path = '/Users/ryszardcetnarski/Desktop/Permutations/Scatter/Independent/'

    #numpy.random.choice(a, size=None, replace=True, p=None)
    n_perm = 10000
    joined= pd.concat([ExtractBands_colstack('before', normed_abs), ExtractBands_colstack('after', normed_abs)], axis = 1)
    befores = joined[band+'_before'].as_matrix()
    afters = joined[band+'_after'].as_matrix()
    #Create a vector of all values, before and after together, later two samples (a new before and new after) will be drawn from it.
    joined_arr = np.concatenate((befores, afters))

    #Get the original values to compare agains permutation resuts
    original_results = CalcDifferencesRatio(befores,afters)
    perm_results = {'coeff': [], 'pval': [], 'sum_ratio':[], 'mean_ratio' : [], 'count_ratio':[], 'avg': []}

    fig = plt.figure()
    fig.suptitle(band + '\nIndependent' )
    orig = fig.add_subplot(111)
  #  drawn = fig.add_subplot(212)

   # orig.hist(joined_arr)
#With replacement  = with duplicates
    replace = False
    for i in range(0,n_perm):
        #random choice returns x by y (size) array with a sample of random values, with or without replacement
        #Using two rows by half a lenght of all data size we get two new, randomly drawn arrays
        perm = np.random.choice(joined_arr, size = (2,int(len(joined_arr)/2)), replace = replace)
        tmp_dict = CalcDifferencesRatio(perm[0,:],perm[1,:])
        for key, value in tmp_dict.items():
            perm_results[key].append(value)
        orig.scatter( perm[1,:] - perm[0,:],perm[0,:],color = 'b', alpha = 0.02, edgecolors='w', s = 100, label = 'permutation' if i == 0 else "")

    PlotPermutation(original_results, perm_results, band, 'Independent', normed_abs)
    orig.scatter( afters - befores,befores,color = 'r', alpha = 0.8, edgecolors='w', s = 100, label = 'original')
    orig.set_xlabel('After - Before')
    orig.set_ylabel('Before')
    orig.legend(loc = 'best')
    fig.savefig(path + band +'_'+ normed_abs+'.png', dpi =300)

    #drawn.hist(perm.T, stacked = True)


def PlotPermutation(orig, perm, band, _type, normed_abs):
    path = '/Users/ryszardcetnarski/Desktop/Permutations/Results/' + _type+'/'

    fig = plt.figure()
    fig.suptitle(band + '\n' + _type, fontweight = 'bold')
    idx = 1
    for measure_name, list_of_results in perm.items():
        ax = fig.add_subplot(230 +idx)
        ax.hist(list_of_results, bins = 15)
        ax.set_title(measure_name)
        ax.axvline(orig[measure_name], color = 'r', linestyle = 'dashed', linewidth = 2)

        ax.tick_params(axis='x', labelsize=5)
        ax.tick_params(axis='y', labelsize=5)

        idx = idx +1
    fig.savefig(path + band+'_'+ normed_abs+'.png', dpi =300)

def CalcDifferencesRatio(x,y):
    #x = x - np.mean(np.concatenate((x,y)))
    #y = y - np.mean(np.concatenate((x,y)))
    diff = y - x
    spadki = np.array([d for d in diff if d <0])
    wzrosty = np.array([d for d in diff if d >0])

    sum_ratio = spadki.sum() / wzrosty.sum()
    mean_ratio = spadki.mean() / wzrosty.mean()
    count_ratio = len(spadki) / len(wzrosty)
    avg = diff.mean()
    coeff, pval = spearmanr(x, diff)

    return {'coeff':coeff, 'pval': pval, 'sum_ratio':sum_ratio, 'mean_ratio' : mean_ratio, 'count_ratio':count_ratio, 'avg': avg}




def PairedPlots():
    fig = plt.figure()
    fig.suptitle('Resting state before after')
    joined= pd.concat([ExtractBands_rowstack('before'), ExtractBands_rowstack('after')], axis = 0, ignore_index=True)
    #joined = joined.transpose()
    #joined['freq'] = joined.index
    #sns.set(style="whitegrid", palette="pastel", color_codes=True)
    # Draw a nested violinplot and split the violins for easier comparison
    sns.violinplot(x ='freqs', y = 'amps', data=joined, hue = 'Before_After',   inner = 'quartiles', split = True, linewidth=1,  palette= "Paired")
    sns.despine(left=True, bottom=True)



#    g = sns.FacetGrid(joined, col="freqs", size=4, aspect=.7)
#    g.map(sns.boxplot, "Before_After", "amps")
#    g.map(sns.stripplot,  "Bcefore_After", "amps", edgecolor = 'white')



    fig = plt.figure()
    ax = sns.boxplot(x='freqs', y='amps', data=joined, hue = 'Before_After')
    ax = sns.stripplot(x='freqs', y='amps', data=joined, jitter = 1, hue = 'Before_After', split = True, palette="Set2")


def ExtractBands_rowstack(before_after, normed_abs):
    SHIFT = 2
    bands_dict =  {'theta':(4,8),'alpha': (8,12), 'smr': (12,15), 'beta1':(15,22), 'beta2':(22,30), 'trained':(12,22)}
    #P3, P4, P5, P6
    selected_electrodes = [18,20,46,49]
    box =  LoadResting(before_after, normed_abs)[:,:,selected_electrodes]
  #  alpha = box[:,bands_dict['alpha'][0] - SHIFT :bands_dict['alpha'][1] - SHIFT +1,:]

    allAmps = []
    freqs = []
    band_vals = pd.DataFrame()
    bef_aft_idx = []
    subject_idx = []

    for band_name, f_range in bands_dict.items():
        #Plus one on the upper index to include the upper frequency boundary
        #First take the sum along the frequency axis(thus eliminating this axis), then take the average across electrodes
        amps = np.mean(np.sum(box[:, f_range[0] - SHIFT -1:f_range[1] - SHIFT ,:], axis =1 ), axis =1)
        allAmps.extend(amps)
        freqs.extend([band_name for i in amps])
        bef_aft_idx.extend([before_after for i in amps])
        #subject_idx.extend()


    band_vals['Before_After'] = bef_aft_idx
    band_vals['amps'] = allAmps
    band_vals['freqs'] = freqs

    return band_vals

def ExtractBands_colstack(before_after, normed_abs):
    """Normally use this one"""
    SHIFT = 2
    bands_dict =  {'theta':(4,8),'alpha': (8,12), 'smr': (12,15), 'beta1':(15,22), 'beta2':(22,30), 'trained':(12,22)}
    #P3, P4, P5, P6
    selected_electrodes = [18,20,46,49]


    box = LoadResting(before_after, normed_abs)[:,:,selected_electrodes]
    #In the resulting array, rows are subjects, columns are electrodes
   # sum_subjec_electrode = np.sum(box, axis = 1)
  #  alpha = box[:,bands_dict['alpha'][0] - SHIFT :bands_dict['alpha'][1] - SHIFT +1,:]

    band_vals = pd.DataFrame()

    for band_name, f_range in bands_dict.items():
        # -2 to correct for the shoft (starts at 2nd herz) and -1 to correct for diefferences between matlab and python, i.e. starting from 1 and including upperbound
        #First take the sum along the frequency axis(thus eliminating this axis), then take the average across electrodes
        #amps =np.mean(np.sum(box[:, f_range[0] - SHIFT -1:f_range[1] - SHIFT ,:], axis =1 ), axis =1)
        amps =np.mean(np.sum(box[:, f_range[0] - SHIFT -1:f_range[1] - SHIFT ,:], axis =1 ), axis =1)
        band_vals[band_name + '_' + before_after] =amps






    return band_vals


#TODO (but first do the permutation) : Change histogram resolution and range in RegressionBeforeAfter(freq)
#TODO make the regression line thinner and perhaps dotted, make span through the whole axis, maybe make the markers different
def SaveAllFig():
    HistRegResiduals('alpha','beta1')
    HistRegResiduals('alpha','beta2')
    HistRegResiduals('alpha','smr')
    HistRegResiduals('alpha','theta')

    HistRegResiduals('beta1','beta2')
    HistRegResiduals('beta1','smr')
    HistRegResiduals('beta1','theta')


    HistRegResiduals('beta2','theta')
    HistRegResiduals('beta2','smr')

    HistRegResiduals('theta','smr')



def HistRegResiduals(freq1,freq2):



    names, conditions, electrodes = LoadRestingInfo()

    joined = pd.concat([ExtractBands_colstack('before', ''), ExtractBands_colstack('after','')], axis  =1)

    joined['conditions'] = conditions
    joined['names'] = names

    subset = joined[[freq1+ "_before", freq1+ "_after", freq2 +"_before", freq2+ "_after", 'conditions']]

    g = sns.PairGrid(subset, hue ='conditions')#,palette="Set2")
#    g.set(ylim=(0, None))
    g.map_upper(sns.regplot)#, scatter_kws = {'edgecolor':"w", 's':40})
    g.map_lower(sns.residplot)#, scatter_kws={'edgecolor':'w', 's':40})
    g.map_diag(plt.hist)

    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45)

    g.add_legend()
    g.set(alpha=0.5)

    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Figury/RestingState/BeforeAfter_Regression/CrossFreq/'
    g.savefig(path +freq1+'_'+freq2+'.png',dpi = 300)
    RegressionBeforeAfter(freq1)
    RegressionBeforeAfter(freq2)
    #return joined
    #for band, subject_rs in before.items()

def RegressionBeforeAfter(freq):
    names, conditions, electrodes = LoadRestingInfo()

    joined = pd.concat([ExtractBands('before'), ExtractBands('after')], axis  =1)
    joined['conditions'] = conditions
    joined['names'] = names

    fig = sns.jointplot(x =freq+ '_before', y=freq+ '_after', data=joined[[freq +'_before', freq +'_after']], kind="reg");

    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Figury/RestingState/BeforeAfter_Regression/InsideFreq/'

    fig.savefig(path +freq +'.png',dpi = 300)

def HeatMaps():
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 5
    plt.rcParams['xtick.direction'] = 'in'
    before = LoadResting('before')
    after = LoadResting('after')
    names, conditions, electrodes = LoadRestingInfo()
    herz = np.around(np.linspace(2,30,28), 0).astype(int)
    herz = [' ' if i%2 == 1 else str(b) for  i,b in enumerate(herz)]
    #electrodes = [' ' if i%2 == 0 else b for  i,b in enumerate(electrodes)]

  #  mask = np.zeros_like(corr, dtype=np.bool)
  #  mask[np.triu_indices_from(mask)] = True
    for i in range(0,44):
        fig = plt.figure()
        fig.suptitle(names[i] + '\n' + conditions[i])
       # ax_bef = fig.add_subplot(221)
       # ax_aft = fig.add_subplot(222)


        ax_bef_log =  plt.subplot2grid((6,5),(0, 0), colspan = 4, rowspan = 3)
        ax_aft_log = plt.subplot2grid((6,5),(3, 0), colspan = 4, rowspan = 3)
        ax_diff = plt.subplot2grid((6,5),(2, 4), colspan = 2, rowspan = 2)

        ax_diff.set_title(' before - after', fontsize =9)

        ax_bef_log.set_ylabel('before')
        ax_aft_log.set_ylabel('after')


        sns.heatmap(np.log10(before[i,:,:]), annot=False, ax = ax_bef_log, center = 0.0, vmin = -1.8, vmax = 1.8,   xticklabels= electrodes,  yticklabels = herz )
        sns.heatmap(np.log10(after[i,:,:]), annot=False,  ax = ax_aft_log, center = 0.0, vmin = -1.8, vmax = 1.8, xticklabels= electrodes,  yticklabels = herz )
        sns.heatmap(np.log10(after[i,:,:]) - np.log10(before[i,:,:]), center = 0.0, vmin = -1.0, vmax = 1.0, annot=False,ax = ax_diff,  xticklabels= False, yticklabels= False)
        #sns.clustermap(after[i,:,:])
        plt.setp( ax_aft_log.xaxis.get_majorticklabels(), rotation=90 )
        plt.setp( ax_bef_log.xaxis.get_majorticklabels(), rotation=90 )
        plt.setp( ax_aft_log.yaxis.get_majorticklabels(), rotation=360 )
        plt.setp( ax_bef_log.yaxis.get_majorticklabels(), rotation=360 )
        plt.savefig('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Figury/RestingState/HeatMaps_Scaled/' + names[i] + '.png', dpi = 300)

    fig = plt.figure()
    fig.suptitle('before average', fontweight = 'bold')
    sns.heatmap(np.log10(np.mean(before, axis =0)), annot=True, annot_kws={"size": 5},fmt='.1f', xticklabels= electrodes,  yticklabels = herz) #ax = ax_bef_log, vmin = -1.6, vmax = 1.6,  xticklabels= False,  yticklabels = herz )

    fig = plt.figure()
    fig.suptitle('after average', fontweight = 'bold')
    sns.heatmap(np.log10(np.mean(after, axis =0)), annot=True, annot_kws={"size": 5},fmt='.1f', xticklabels= electrodes,  yticklabels = herz) #ax = ax_bef_log, vmin = -1.6, vmax = 1.6,  xticklabels= False,  yticklabels = herz )

    fig = plt.figure()
    fig.suptitle('average difference', fontweight = 'bold')
    sns.heatmap(np.log10(np.mean(after, axis =0)) - np.log10(np.mean(before, axis =0)),  annot=True, annot_kws={"size": 5},fmt='.1f', xticklabels= electrodes,  yticklabels = herz) #ax = ax_bef_log, vmin = -1.6, vmax = 1.6,  xticklabels= False,  yticklabels = herz )

#    fig = plt.figure()
#    fig.suptitle('cumulative sum', fontweight = 'bold')
#    sns.heatmap(np.log10(np.sum(after, axis =0)) + np.log10(np.sum(before, axis =0)),  annot=True, annot_kws={"size": 5},fmt='.1f', xticklabels= electrodes,  yticklabels = herz) #ax = ax_bef_log, vmin = -1.6, vmax = 1.6,  xticklabels= False,  yticklabels = herz )

def LoadResting(before_after, normed_abs):
    '''If normed leave normed_abs empty, ("")'''
    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Jacek_RestingState/'+ normed_abs+before_after+'_fft.mat'
    mat_struct = sio.loadmat(path)[before_after +'_fft']
    mat_struct = np.swapaxes(mat_struct, 0,2)
    mat_struct = np.swapaxes(mat_struct, 1,2)
    return mat_struct

def PlotAll():
    before, after = LoadResting('before', 'normed_')[:,:,18], LoadResting('after', 'normed_')[:,:,18]

    for sub_before, sub_after in zip(before, after):
        fig = plt.figure()
        absolute = fig.add_subplot(211)
        relative= fig.add_subplot(212)
        absolute.plot(sub_before, 'b', label = 'before')
        absolute.plot(sub_after, 'r', label = 'after')
        relative.plot(sub_before / np.sum(sub_before), 'b')
        relative.plot(sub_after / np.sum(sub_after),'r')
        absolute.legend(loc = 'best')

        absolute.set_ylabel('absolutne wartosci')
        relative.set_ylabel('podzielone prez sume')
def LoadRestingInfo():

    with open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Jacek_RestingState/names.csv') as f:
        reader = csv.reader(f)
        names = list(reader)[0]

    with open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Jacek_RestingState/conditions.csv') as f:
        reader = csv.reader(f)
        conditions = list(reader)[0]

    with open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Jacek_RestingState/electrodes.csv') as f:
        reader = csv.reader(f)
        electrodes = list(reader)[0]

    return names, conditions, electrodes

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


def CheckSingleSubject():
   # selected_electrodes = [18,20,46,49]
    #bands_dict =  {'theta':(4,8),'alpha': (8,12), 'smr': (12,15), 'beta1':(15,22), 'beta2':(22,30), 'trained':(12,22)}
    trained_avg = []
    all_subs= LoadResting('before')
    _min =12
    _max =20

    for i in range(0,44):
        sub = all_subs[i,:,[18,20,46,49]].T
        #Suma na wszystkich elektrodach
        #electrode_sum = np.sum(sub, axis = 0)

        trained = sub[12:20,:]

        trained_sum = np.sum(trained, axis =0)


       # trained_normed = trained_sum/electrode_sum

       # trained_avg.append(np.mean(trained_normed))
        trained_avg.append(np.mean(trained_sum))

    x = np.array(trained_avg)
    plt.figure()
    plt.plot(x,y, 'o')
    plt.xlim(0.05, 0.3)
    plt.ylim(0.05, 0.3)

    return x