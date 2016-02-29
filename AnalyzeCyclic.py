# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:31:50 2016

@author: ryszardcetnarski
Loads Database from Prepare_DB_Base_Train and continues from there

Pasma in DB: ['all_spectrum', 'theta', 'alpha', 'smr', 'beta1', 'beta2','beta3', 'trained','ratio']
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import Prepare_DB_Base_Train as prep
from scipy.stats.kde import gaussian_kde



def PlotTrainingInTime():
    path = '/Users/ryszardcetnarski/Desktop/By_time/'
    db = LoadDatabase()#.iloc[0:87]
    plt.style.use('ggplot')
    for name, subject in db.groupby(db.index):
        fig = plt.figure()
        ax =fig.add_subplot(111)
        print(name)
        training = ExtractBands(subject, 'training', 'all_spectrum')
        baseline = ExtractBands(subject.dropna(subset = ['baseline_bands']), 'baseline', 'all_spectrum')
        ax.plot(subject['days_from_first'], training, color = 'green', label = 'training')
        ax.plot(subject.dropna(subset = ['baseline_bands'])['days_from_first'],baseline, color = 'orange', label = 'baseline')
        ax.set_xlabel('ilość dni od pierwszego treningu')
        ax.set_ylabel('all_spectrum')
        ax.legend(loc = 'best')
        fig.savefig(path +name +'.png', dpi = 300)
        return training, baseline

def CorrelateTrainingBaseline():
    db = LoadDatabase()#.iloc[0:87]
    plt.style.use('ggplot')
    base_train = {'baseline':[], 'training':[] }
    for name, subject in db.groupby(db.index):

        training = Z_score(ExtractBands(subject.dropna(subset = ['baseline_bands']), 'training', 'all_spectrum'))
        baseline = Z_score(ExtractBands(subject.dropna(subset = ['baseline_bands']), 'baseline', 'all_spectrum'))
        base_train['baseline'].extend(baseline)
        base_train['training'].extend(training)

    #fig = plt.figure()
    #ax =fig.add_subplot(111)

    base_train = pd.DataFrame(base_train)
    sns.jointplot("baseline", "training", data=base_train, kind="reg", color="r", size=7)
    return base_train

#    reduced_db = pd.DataFrame([ExtractBands(db, 'training', 'alpha'), db['days_from_first'].values]).transpose()
#    reduced_db['subject'] = db.index
#    reduced_db.columns =  ['amp', 'days_from_start', 'subject']
#
#        # Initialize a grid of plots with an Axes for each walk
#    grid = sns.FacetGrid(reduced_db, col='subject', hue='subject',)
#
#    # Draw a horizontal line to show the starting point
#    #grid.map(plt.axhline, y=0, ls=":", c=".5")
#
#    # Draw a line plot to show the trajectory of each random walk
#    grid.map(plt.plot, 'days_from_start', 'amp', marker="o")#, ms=4)
#
#    # Adjust the arrangement of the plots
#    grid.fig.tight_layout()#w_pad=1)

   # return reduced_db

    #fig = plt.

def PlotIndivisualDistributions():
    path = '/Users/ryszardcetnarski/Desktop/Distributions/'
    plt.style.use('ggplot')
    db = LoadDatabase()
    rest = prep.Load_rest()
    kde_bandwith = 0.8

    #dist_func = gaussian_kde(initial, )


    #Vector for plotting
    for name, subject in db.groupby(db.index):
        fig = plt.figure()
        fig.suptitle(name, fontweight ='bold')

        bands = ['all_spectrum', 'alpha', 'beta1', 'beta2']
        ax = []
        for idx,band in enumerate(bands):
            ax.append(fig.add_subplot(220+idx+1))

            training = ExtractBands(subject, 'training', band)
            baseline = ExtractBands(subject.dropna(subset = ['baseline_bands']), 'baseline', band)

            training_distribution = gaussian_kde(training, kde_bandwith)
            baseline_distribution = gaussian_kde(baseline, kde_bandwith)

            ax[idx].hist(training, alpha = 0.2, normed = True, color = 'blue')
            ax[idx].hist(baseline, alpha = 0.2, normed = True, color = 'yellow')


            if name in rest:
                ax[idx].axvline(rest[name]['Before'].loc[band].mean(), color = 'b', linestyle = 'dashed', linewidth = 2, label = 'rest przed')
                ax[idx].axvline(rest[name]['After'].loc[band].mean(), color = 'r', linestyle = 'dashed', linewidth = 2, label = 'rest po')
            else:
                print(name)

            xmin, xmax = ax[idx].get_xlim()
            x = np.linspace(xmin-1, xmax+1, 100)

            ax[idx].plot(x, training_distribution(x), color = 'blue', label ='dystrybucja trening')
            ax[idx].plot(x, baseline_distribution(x), color = 'yellow', label ='dystrybucja baseline')

            ax[idx].set_title(band)
            if(idx == 3):
                ax[idx].legend(loc = 'best')

        fig.savefig(path + name +'.png', dpi = 400)

        #break
       #rest.loc[name]['Before'].loc['alpha'])
       # return rest.loc[name]['Before'].loc['alpha']
    plt.tight_layout()


def LoadDatabase():
    df = pd.read_pickle('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/DatabaseTrainBase.pkl')
    return df



def ExtractBands(db, train_base, band):
    all_sessions= db[train_base +'_bands']
    all_session_one_band = []
    for session in all_sessions:
        all_session_one_band.append(np.mean(session.loc[band]))
    return np.array(all_session_one_band)


def Z_score(vector):
##Sanity check, should be a numpy array anyways. If not np, then subtraction might not subtract a constant from all elements
    copy = vector
    nan_idx = np.isnan(vector)
    #Compute zscore for non nans
    vector = vector[~nan_idx]
    z_score = (vector - np.mean(vector))/np.std(vector)
    #Substitute non nans with z score and keep the remaining nans
    copy[~nan_idx] = z_score
    return copy
