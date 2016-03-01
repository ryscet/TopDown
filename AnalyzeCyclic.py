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

def RunAllGrouped():
    bands = ['all_spectrum', 'alpha', 'beta1', 'beta2', 'smr','ratio', 'theta', 'trained', 'ratio']
    for band in bands:
        #GroupedDistributionsPlot(band)
        GroupbyProtocolDistributionsPlot(band)



def AllTogetherDistributionsPlot(band):
    path = '/Users/ryszardcetnarski/Desktop/GroupedDistributions/'
    plt.style.use('seaborn-bright')
    db = LoadDatabase()
    rest = prep.Load_rest()
    kde_bandwith = 0.8

    #dist_func = gaussian_kde(initial, )

    #axes for plotting
    fig = plt.figure()
    #ax = [fig.add_subplot(220+idx+1) for idx, band in enumerate(bands)]
    ax = fig.add_subplot(111)

    all_training = []
    all_baseline = []

    all_rest_after = []
    plot_lims = {'all_spectrum': (-60,60), 'alpha': (-10,10), 'beta1': (-20,20), 'beta2': (-20,20),
    'beta3':(-20,20), 'smr':(-5,5),'ratio': (-1.2,1.2), 'theta':(-10,10), 'trained':(-15,15) }
  #  i is just for labels not to duplicate
    i = 0
    for name, subject in db.groupby(db.index):

        if name in rest:

            #Name is subject, .mean() is taken from the four training electrodes

            rest_before = rest[name]['Before'].loc[band].mean()

            training = ExtractBands(subject, 'training', band)
            all_training.extend(training - rest_before )

            baseline = ExtractBands(subject.dropna(subset = ['baseline_bands']), 'baseline', band)
            all_baseline.extend(baseline - rest_before)

            training_distribution = gaussian_kde(training - rest_before, kde_bandwith)
            baseline_distribution = gaussian_kde(baseline - rest_before, kde_bandwith)



            ax.axvline(rest[name]['After'].loc[band].mean() - rest_before, color = 'black',alpha = 0.2, linestyle = 'dashed', linewidth = 2, label = 'rest po' if i == 0 else "")

            all_rest_after.append(rest[name]['After'].loc[band].mean() - rest_before)


           # ax.set_xlim(-50, 50)
           # ax.set_ylim(0, 0.2)
            xmin, xmax = plot_lims[band]
            #xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 1000)
#
            ax.plot(x , training_distribution(x), color = 'blue', alpha = 0.3, label ='dystrybucja trening'  if i == 0 else "")
            ax.plot(x , baseline_distribution(x), color = 'green',alpha = 0.3, label ='dystrybucja baseline'  if i == 0 else "")

            ax.set_title(band)
            #Iterator only used for labels
            i = i +1
             #Dirty hack
            if(band == 'alpha'):
                ax.set_ylim(0,1)

#            if(idx == 3):
#                ax[idx].legend(loc = 'best')
#            print(xmin)
#            #return

           # fig.savefig(path + name +'.png', dpi = 400)

            #break
           #rest.loc[name]['Before'].loc['alpha'])
           # return rest.loc[name]['Before'].loc['alpha']
        else:
            print('Not in rest: '+name)
    ax.axvline(0, color = 'red', linestyle = 'dashed', linewidth = 3,alpha = 0.9, label = 'rest przed')


    all_training = np.array(all_training)
    all_baseline = np.array(all_baseline)

    train_group_ditribution = gaussian_kde(all_training, kde_bandwith)
    base_group_ditribution = gaussian_kde(all_baseline, kde_bandwith)

    ax.plot(x , train_group_ditribution(x) *5, color = 'blue', alpha = 0.8)
   # ax.plot(x , base_group_ditribution(x) *5, color = 'green', alpha = 0.8)

    ax.axvline(np.array(all_rest_after).mean(), color = 'black', linestyle = 'dashed', linewidth = 2,alpha = 0.5)

    ax.set_xlim(plot_lims[band])

    ax.legend()

    plt.tight_layout()
    ax.grid(b=False)
    fig.savefig(path+ '_' + band +'.png', dpi =300)


def IndivisualDistributionsPlot():
    path = '/Users/ryszardcetnarski/Desktop/Distributions/'
    plt.style.use('ggplot')
    db = LoadDatabase()
    rest = prep.Load_rest()
    kde_bandwith = 0.8



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

            ax[idx].hist(training , alpha = 0.2, normed = True, color = 'blue')
            ax[idx].hist(baseline , alpha = 0.2, normed = True, color = 'yellow')


            if name in rest:
                ax[idx].axvline(rest[name]['Before'].loc[band].mean(), color = 'b', linestyle = 'dashed', linewidth = 2, label = 'rest przed')
                ax[idx].axvline(rest[name]['After'].loc[band].mean(), color = 'r', linestyle = 'dashed', linewidth = 2, label = 'rest po')
                # ax[idx].axvline(0, color = 'b', linestyle = 'dashed', linewidth = 2, label = 'rest przed')
                # ax[idx].axvline(0, color = 'r', linestyle = 'dashed', linewidth = 2, label = 'rest po')

            else:
                print(name)

            xmin, xmax = ax[idx].get_xlim()
            x = np.linspace(xmin-1, xmax+1, 100)
#
            ax[idx].plot(x , training_distribution(x), color = 'blue', label ='dystrybucja trening')
            ax[idx].plot(x , baseline_distribution(x), color = 'yellow', label ='dystrybucja baseline')

            ax[idx].set_title(band)
            if(idx == 3):
                ax[idx].legend(loc = 'best')



        fig.savefig(path + name +'.png', dpi = 400)

        #break
       #rest.loc[name]['Before'].loc['alpha'])
       # return rest.loc[name]['Before'].loc['alpha']
    plt.tight_layout()



def GroupbyProtocolDistributionsPlot(band):
    path = '/Users/ryszardcetnarski/Desktop/GroupedDistributions/'
    plt.style.use('seaborn-bright')
    db = LoadDatabase()
    rest = prep.Load_rest()
    kde_bandwith = 0.8

    #dist_func = gaussian_kde(initial, )




    plot_lims = {'all_spectrum': (-60,60), 'alpha': (-30,30), 'beta1': (-40,40), 'beta2': (-40,40),
    'beta3':(-40,40), 'smr':(-15,15),'ratio': (-1.2,1.2), 'theta':(-20,20), 'trained':(-15,15) }
  #  i is just for labels not to duplicate
    i = 0
    db['condition'] = db['condition'].str.replace('plus', 'mixed')
    db['condition'] = db['condition'].str.replace('minus', 'mixed')

    fig = plt.figure()
    gi = 1
    for group_name, condition_group in db.groupby('condition'):
          #axes for plotting
        #ax = [fig.add_subplot(220+idx+1) for idx, band in enumerate(bands)]
        ax = fig.add_subplot(120+gi)
        all_training = []
        all_baseline = []

        all_rest_after = []
        rest_diffs = []
        gi = gi+1
        for name, subject in condition_group.groupby(condition_group.index):
            if name in rest:

                #Name is subject, .mean() is taken from the four training electrodes

                rest_before = rest[name]['Before'].loc[band].mean()
                #Rest after relative to rest before
                rest_after = rest[name]['After'].loc[band].mean() - rest_before
                training = ExtractBands(subject, 'training', band)
                all_training.extend(training - rest_before )

                baseline = ExtractBands(subject.dropna(subset = ['baseline_bands']), 'baseline', band)
                all_baseline.extend(baseline - rest_before)

                training_distribution = gaussian_kde(training - rest_before, kde_bandwith)
                baseline_distribution = gaussian_kde(baseline - rest_before, kde_bandwith)



                ax.axvline(rest_after, color = 'black',alpha = 0.2, linestyle = 'dashed', linewidth = 2, label = 'rest po' if i == 0 else "")

                all_rest_after.append(rest[name]['After'].loc[band].mean() - rest_before)

                #Code wzrosty i spadki
                if(rest_after < 0):
                    rest_diffs.append(-1)
                else:
                    rest_diffs.append(1)

               # ax.set_xlim(-50, 50)
               # ax.set_ylim(0, 0.2)
                xmin, xmax = plot_lims[band]
                #xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 1000)
    #
                ax.plot(x , training_distribution(x), color = 'blue', alpha = 0.3, label ='dystrybucja trening'  if i == 0 else "")
                ax.plot(x , baseline_distribution(x), color = 'green',alpha = 0.3, label ='dystrybucja baseline'  if i == 0 else "")

                ax.set_title(group_name + ' ' +band)
                #Iterator only used for labels
                i = i +1
                 #Dirty hack
                if(band == 'alpha'):
                    ax.set_ylim(0,1)

    #            if(idx == 3):
    #                ax[idx].legend(loc = 'best')
    #            print(xmin)
    #            #return

               # fig.savefig(path + name +'.png', dpi = 400)

                #break
               #rest.loc[name]['Before'].loc['alpha'])
               # return rest.loc[name]['Before'].loc['alpha']
            else:
                print('Not in rest: '+name)
        ax.axvline(0, color = 'red', linestyle = 'dashed', linewidth = 3,alpha = 0.9, label = 'rest przed')

        n_spadki = len([i for i in rest_diffs if i <0])
        n_wzrosty = len([i for i in rest_diffs if i >0])
        ax.annotate( 'wzrosty: '  +str(n_wzrosty) +' spadki: '+str(n_spadki) , xy =(xmin, 0.05))

        all_training = np.array(all_training)
        all_baseline = np.array(all_baseline)

        train_group_ditribution = gaussian_kde(all_training, kde_bandwith)
        base_group_ditribution = gaussian_kde(all_baseline, kde_bandwith)

        ax.plot(x , train_group_ditribution(x) *5, color = 'blue', alpha = 0.8)
       # ax.plot(x , base_group_ditribution(x) *5, color = 'green', alpha = 0.8)

        ax.axvline(np.array(all_rest_after).mean(), color = 'black', linestyle = 'dashed', linewidth = 2,alpha = 0.5)

        ax.set_xlim(plot_lims[band])

        ax.legend()

        plt.tight_layout()
        ax.grid(b=False)
      #  fig.savefig(path+ '_' + band +'.png', dpi =300)


def LoadDatabase():
    df = pd.read_pickle('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/DatabaseTrainBaseUPDATED.pkl')
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
