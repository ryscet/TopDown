# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:45:23 2016

@author: ryszardcetnarski
"""

from sklearn.decomposition import PCA as sklearnPCA


from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn.preprocessing import StandardScaler
import os
import glob
import scipy.io as sio
import numpy as np
import pandas as pd

#TODO maybe add a condition that only blocks with a lower mean from the groupo mean can be filtered out
#Then also add manually marking artifacts because for fuck sake there is more of them then inliers- so maybe treat the good ones as outliers
def Run():
    labels = FilterIndividualAndGroup()
    MarkOutliers(labels[labels['mask'] ==1])


def MarkOutliers(bad_ones):
    """bad ones is a data frame with a column alltogehter wich contains strings made from name+block+subject
    which represent blocks where pca picked them as outliers. First run Individual pca filter which makes a dataase
    of all subjects session and blocks and a column which discriminates inliers and outliers pd(columns = [subject, session, block, alltogether, mask])
    call like MarkBadOnes(labels[labels['mask'] ==1])"""


    allFFT, freqs = Load_FFT(2,50)
    for name, fft in allFFT.items():
        fig= plt.figure()
        fig.suptitle(name, fontweight = 'bold')
        ax = fig.add_subplot(111)

        print('plotting %s' %name)
        for session in range(0, fft.shape[0]):

            for block in range(0,fft.shape[2]):

                single_fft = fft[session,:,block][~np.isnan(fft[session,:,block])]
                if(np.count_nonzero(~np.isnan(single_fft)) > 1):
                    color = 'blue'
                    if(bad_ones['alltogether'].str.contains(name +str(session)+str(block)).any()):
                        color = 'red'
                    ax.plot(freqs, np.log(single_fft),  color = color, alpha = 0.2)



def FilterIndividualAndGroup():
    """This function returns all blocks annotated as inliers or outliers.
    First it runs PCA on blocks grouped by individuals and identifies outliers based on malahanobis distance
    of each block 1st and 2nd component.
    Then it runs pca on all blocks combined and clusters them into 2 clusters, filtering out the smaller one"""

    #Structure for keeping the information about in/outliers (miesniowcy)
    labels = pd.DataFrame(data = [[None, None,None, None, None]],columns = ('subject', 'session', 'block', 'alltogether', 'mask'))
    #loads a  dict of subjects, each subject is a list of arrays, each array are blocks*session ( Session, signalIdx, block)
    allFFT, freqs = Load_FFT(15, 40)
    #Indexer just for labels df
    df_idx = 0
    all_labels = []
    for name, fft in allFFT.items():
        subject_fft = []

        for session in range(0, fft.shape[0]):
            #return fft[session,:,:]
            for block in range(0,fft.shape[2]):
                single_fft = fft[session,:,block][~np.isnan(fft[session,:,block])]
                #Check for all Nan sessions
                if(np.count_nonzero(~np.isnan(single_fft)) > 1):
                    #Store info in labels
                    labels.loc[df_idx] = [name, session, block, name+str(session)+str(block), None]
                    df_idx = df_idx+1
                    #store non nan blocks in hnia
                    subject_fft.append(np.log(np.array(single_fft)))
        #Perform pca based on all blocks from subject to filter out outliers
        all_labels.extend( PcaFilter(np.array(subject_fft), name, False, 'outlier'))
        #print(np.array(subject_fft).shape)
        if('all_fft' in locals()):
            all_fft = np.vstack((all_fft, subject_fft))

        else:
            all_fft = np.array(subject_fft)
    all_labels = np.array(all_labels)
    #print('N outliers inside individual: %i' %sum(all_labels))

        #Put pca results in info
#Do the second PCA filtering outliers on individual level
    #print('all blocks: %d' %len(all_fft))
    filtered = all_fft[np.where(all_labels == 0)[0],:]
    #print('filtered: %d' %len(filtered))

    grouped_labels = PcaFilter(filtered, 'all subjects',True, 'cluster')
   # print('N outliers across group: %i' %sum(grouped_labels))

    #Combine the score from individual and group filtering
    all_labels[np.where(all_labels == 0)[0]] = all_labels[np.where(all_labels == 0)[0]] + grouped_labels
    labels['mask']= all_labels

    return labels


def PcaFilter(X, name, PLOT_ON, method):
    """Individual subjects have outlier blocks filtered out based on malahanobis distance of their pca 1st and 2nd component
    (each subject is projected into their own variance space).
    Returns mask array (of 0's and 1's lenght of X) indicing bad and good blocks"""
    #The convention for mask arrays is 0 - inlier, 1 - outlier

    #Pca decomposition into first two components
    sklearn_pca = sklearnPCA(n_components=2)
    pcs = sklearn_pca.fit_transform(X)


    if(method == 'outlier'):
        #This index corresponds to the original index on the array of time series
        #last argument is the threshold of how many standard deviations away a point is considered an outlier
        outlier_idx = MD_removeOutliers(pcs[:,0], pcs[:,1], 2)
        #this will be used for a boolean array for filtering.
        mask_array = np.zeros(len(pcs))
        if(len(outlier_idx) >0 ):
            mask_array[outlier_idx] = 1


    if (method == 'cluster'):
        mask_array = Cluster(pcs)

    if(PLOT_ON):
        colors = ['r', 'b']
        fig = plt.figure()
        fig.suptitle(name)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        #Plot PCA scores and mark outliers
        ax2.scatter(pcs[:,0], pcs[:,1], c = mask_array, cmap = 'jet', s = 60, marker = 'o')
        #Print variance ratio
        ax2.annotate(sklearn_pca.explained_variance_ratio_,xy= (1,1), xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')
        #Plot original signals and mark PCA indentified outliers
        for idx,row in enumerate(X):
            ax1.plot(row, color =colors[int(mask_array[idx])], alpha = 0.2)

    return mask_array




def MD_removeOutliers(x, y, std):
    #Std - how many standard deviations avay from the mean to exclude
    MD = MahalanobisDist(x, y)
    threshold = np.mean(MD)+np.std(MD)*std # adjust 1.5 accordingly
    nx, ny, outliers = [], [], []
    for i in range(len(MD)):
        if MD[i] <= threshold:
            nx.append(x[i])
            ny.append(y[i])
        else:
            outliers.append(i) # position of removed pair
    return np.array(outliers)


def MahalanobisDist(x, y):
    covariance_xy = np.cov(x,y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])

    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    return md



def Cluster(X):
    range_n_clusters = [2]
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=2, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])



        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhoutte score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("1st principal component")
        ax2.set_ylabel("2nd principal component")

        plt.suptitle(("Silhouette analysis for KMeans clustering on pca scores "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()

        #Clusters will switch labels depending on random initialization
        #If there are mostly ones - i.e. inliers are marked as 1, reverse
    if(np.count_nonzero(cluster_labels.astype(int)) > len(cluster_labels.astype(int))/2 ):
        #reverse 0's and 1's
        mask_array = np.zeros(len(cluster_labels.astype(int))) + 1 - cluster_labels.astype(int)
    else:
        mask_array = cluster_labels.astype(int)


    return mask_array


def Load_FFT(min_freq,max_freq ):
    '''train_base_rest = train | base | rest'''
#TODO check whats up with the overlapping subjects from tura 2 and 3, repeat when normalization amplitude methds are finally decided (divide by sum/mean, take sum/mean)
    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/fft_treningi/'


    #Select only file names without proc suffix (procenty) and removing the electrode and Abs_amp prefix
    names = [os.path.basename(x)[11:-4] for x in glob.glob(path+'*') if 'P4' in x]
    #Need to filter for proc again, to pass it for later subject selection, otherwise names uniques will include procs
    full_paths = [x for x in glob.glob(path+'*')]
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
            all_electrodes.append(sio.loadmat(file)['spectra'].swapaxes(0,2).swapaxes(1,2))
            #Load freqs only once
            if('freqs' not in locals()):
                #Get the indexes where the frequencies are in bound of those of interest. Index does not eqal frequency
                freqs = sio.loadmat(file)['freqs']
                min_idx = np.where(freqs> min_freq)[0][0]
                max_idx = np.where(freqs> max_freq)[0][0]
                freqs = freqs[min_idx:max_idx]
            all_subjects[subject] = AverageElectrodes(all_electrodes)[:,min_idx:max_idx,:]#only take the first 30 hz

    return all_subjects, freqs



def AverageElectrodes(all_electrodes):

#Empty array in the shape of the first element
    _sum = np.zeros(all_electrodes[0].shape)
#Sum and divide by n elements
    for electrode in all_electrodes:
        _sum = _sum + electrode
    avg = _sum / len(all_electrodes)

    return avg



