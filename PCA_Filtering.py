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
import seaborn as sns

def PCA(X, labels):

    sklearn_pca = sklearnPCA(n_components=2)
    Y_sklearn = sklearn_pca.fit_transform(X)

    fig = plt.figure()
   # fig.suptitle(band, fontweight = 'bold')
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    km = KMeans(n_clusters=2)
    km.fit(Y_sklearn)


   #ax.scatter(Y_sklearn[:,0], Y_sklearn[:,1])

    #tresh = [0 if i >4 else 1 for i  in Y_sklearn[:,0]]
    colors = ['blue', 'red']
    for idx, row in enumerate(Y_sklearn):
        ax2.scatter(row[0], row[1], color =colors[km.labels_[idx]], alpha = 0.5)
        ax1.plot(X[idx,:], color = colors[km.labels_[idx]], alpha = 0.2)

    labels.iloc[km.labels_ ==1].to_csv('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/miesniowcy_pca.csv ')
    return km.labels_, labels.iloc[km.labels_ ==1]









def PlotMeans():
    allFFT, freqs = Load_FFT()

    for name, fft in allFFT.items():
        fig= plt.figure()
        fig.suptitle(name, fontweight = 'bold')
        ax = fig.add_subplot(111)
        for session in range(0, fft.shape[0]):
            #return fft[session,:,:]
           # for block in range(0,fft.shape[2]):
            single_session =fft[session,:,:]
            if(np.count_nonzero(~np.isnan(single_session)) > 1):

                #single_fft = np.mean(fft[session,:,block][~np.isnan(fft[session,:,block])], axis = 2)
                mean_fft = np.mean(single_session, axis = 1)[:, np.newaxis]
                ax.plot(freqs,mean_fft, alpha = 0.1)
                #return mean_fft, freqs

def PlotMothafuckingAll(bad_ones = None):
    #Dirty hack, because during the first run of pca resultsfor filtering are
    #not knownthis function also creates the bad ones, so after first run it can be fed back to it
    if(bad_ones == None):
        bad_ones = pd.DataFrame(data=['0'],columns = ['alltogether'])
    allFFT, freqs = Load_FFT()
#    fig= plt.figure()
#    all_fft_pca = []
#  #  fig.suptitle(name, fontweight = 'bold')
#    ax = fig.add_subplot(111)
    labels = pd.DataFrame(columns = ('subject', 'session', 'block', 'alltogether'))
    i = 0

    all_fft_pca = []

    for name, fft in allFFT.items():
        fig= plt.figure()
        fig.suptitle(name, fontweight = 'bold')
        ax = fig.add_subplot(111)


        for session in range(0, fft.shape[0]):
            #return fft[session,:,:]
            for block in range(0,fft.shape[2]):

                single_fft = fft[session,:,block][~np.isnan(fft[session,:,block])]
                if(np.count_nonzero(~np.isnan(single_fft)) > 1):
                    color = 'blue'
                    if(bad_ones['alltogether'].str.contains(name +str(session)+str(block)).any()):
                        color = 'red'
                    ax.plot(freqs,np.log(single_fft), alpha = 0.5, color = color)
                    all_fft_pca.append(np.log(single_fft))

                    labels.loc[i] = [name,session, block, name+str(session)+str(block)]
                    i = i+1

    return np.array(all_fft_pca), labels


def IndividualPcaFilter():
    """This function returns only non nan blocks in a very long list of all blocks.
    Secondly it returns a very long dataframe which describe each index of the fft list with subject, session, block info,
    also stating if it was identified as an outlier block using pca+malahanobis (mask column in labels)"""

    #Structure for keeping the information about in/outliers (miesniowcy)
    labels = pd.DataFrame(data = [[None, None,None, None, None]],columns = ('subject', 'session', 'block', 'alltogether', 'mask'))
    #loads a  dict of subjects, each subject is a list of arrays, each array are blocks*session ( Session, signalIdx, block)
    allFFT, freqs = Load_FFT()
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
        print(np.array(subject_fft).shape)
        if('all_fft' in locals()):
            all_fft = np.vstack((all_fft, subject_fft))

        else:
            all_fft = np.array(subject_fft)
    all_labels = np.array(all_labels)

    print('N outliers inside individual: %f' %sum(all_labels))

        #Put pca results in info
#Do the second PCA filtering outliers on individual level
    print('all blocks: %d' %len(all_fft))
    filtered = all_fft[np.where(all_labels == 0 )[0],:]
    print('filtered: %d' %len(filtered))

    grouped_labels = PcaFilter(filtered, 'all subjects',True, 'cluster')
    print('N outliers across group: %f' %sum(grouped_labels))

    #Combine the score from individual and group filtering
    all_labels[np.where(all_labels == 0 )[0]] = all_labels[np.where(all_labels == 0 )[0]] + grouped_labels
    labels['mask']= all_labels

    return labels#, all_fft


def PcaFilter(X, name, PLOT_ON, method):
    print('Inside PCA filter')
    """Individual subjects have outlier blocks filtered out based on malahanobis distance of their pca 1st and 2nd component
    (each subject is projected into their own variance space).
    Returns mask array (of 0's and 1's lenght of X) indicing bad and good blocks"""
    #Pca decomposition into first two components
    sklearn_pca = sklearnPCA(n_components=2)
    pcs = sklearn_pca.fit_transform(X)

    if(method == 'outlier'):
        #This index corresponds to the original index on the array of time series
        outlier_idx = MD_removeOutliers(pcs[:,0], pcs[:,1])
        #this will be used for a boolean array for filtering.
        mask_array = np.zeros(len(pcs))
#The convention for mask arrays is 0 - inlier, 1 - outlier
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




def MD_removeOutliers(x, y):
    MD = MahalanobisDist(x, y)
    threshold = np.mean(MD)+np.std(MD)*3 # adjust 1.5 accordingly
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

   # fig = plt.figure()
   # fig.suptitle(band, fontweight = 'bold')
   # ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(312)
    #ax3 = fig.add_subplot(313)



   # ax1.scatter(Y_sklearn[:,0], Y_sklearn[:,1], alpha = 0.5)
  #  ax = sns.kdeplot(Y_sklearn[:,0], Y_sklearn[:,1], ax = ax1)

#g = (sns.jointplot("sepal_length", "sepal_width", data=iris, color="k").plot_joint(sns.kdeplot, zorder=0, n_levels=6))
   # g = (sns.jointplot(Y_sklearn[:,0], Y_sklearn[:,1], kind="kde", stat_func=None,).set_axis_labels("x", "y"))
   # g = (sns.jointplot(Y_sklearn[:,0], Y_sklearn[:,1]).plot_joint(sns.kdeplot, zorder=0, n_levels=6))
    ###############################################################################


    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

def Cluster(X):
    range_n_clusters = [2]

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

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

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
        return cluster_labels.astype(int)
        # Display results

    # Show data set


#    #tresh = [0 if i >4 else 1 for i  in Y_sklearn[:,0]]
#    colors = ['blue', 'red']
#    for idx, row in enumerate(Y_sklearn):
#        ax2.scatter(row[0], row[1], color =colors[km.labels_[idx]], alpha = 0.5)
#        ax1.plot(X[idx,:], color = colors[km.labels_[idx]], alpha = 0.2)
#
#    #labels.iloc[km.labels_ ==1].to_csv('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/miesniowcy_pca.csv ')
#
#    return km.labels_, labels.iloc[km.labels_ ==1]







def Load_FFT():
    '''train_base_rest = train | base | rest'''
#TODO check whats up with the overlapping subjects from tura 2 and 3, repeat when normalization amplitude methds are finally decided (divide by sum/mean, take sum/mean)
    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/fft_treningi/'

#Divide by 2 to get the actual frequency in Hz (more or less, +-1)
    min_freg = 25
    max_freq = 65
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
                freqs = sio.loadmat(file)['freqs'][min_freg:max_freq]
            all_subjects[subject] = AverageElectrodes(all_electrodes)[:,min_freg:max_freq,:]#only take the first 30 hz
            #all_subjects[subject] = all_electrodes#[0][:,0:30,:]#only take the first 30 hz
    return all_subjects, freqs



def AverageElectrodes(all_electrodes):

#Empty array in the shape of the first element
    _sum = np.zeros(all_electrodes[0].shape)
#Sum and divide by n elements
    for electrode in all_electrodes:
        _sum = _sum + electrode
    avg = _sum / len(all_electrodes)

    return avg


#
#
#    #km = KMeans(n_clusters=2)
#    #km.fit(Y_sklearn)
## generate the linkage matrix
#    Z = linkage(Y_sklearn , 'ward')
#    # calculate full dendrogram
#
#    ax2.set_title('Hierarchical Clustering Dendrogram')
#    ax2.set_xlabel('sample index')
#    ax2.set_ylabel('distance')
#    ax1.set_title( str(sklearn_pca.explained_variance_ratio_))
#    dendrogram(
#        Z,
#        truncate_mode='lastp',  # show only the last p merged clusters
#        p=12,  # show only the last p merged clusters
#        leaf_rotation=90.,
#        leaf_font_size=12.,
#        show_contracted=True,
#        ax = ax2# to get a distribution impression in truncated branches
#    )
#
#    k=3
#   clusters = fcluster(Z, k, criterion='maxclust')
# #   clusters=fcluster(Z, 8, depth=12)
#
#
#    ax1.scatter(Y_sklearn[:,0], Y_sklearn[:,1], c=clusters, cmap='jet')
#    color = ['','r','b','g']
#    for idx,row in enumerate(X):
#        ax3.plot(row, c = color[clusters[idx]], alpha =0.2)
