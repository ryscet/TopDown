from scipy.stats.kde import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import RestingState as rs

from scipy import signal

from scipy.stats import pearsonr
from scipy.misc import derivative
from sklearn.neighbors import KernelDensity
from scipy.stats import expon
import scipy.stats
# creating data with two peaks
#sampD1 = norm.rvs(loc=-1.0,scale=1,size=300)
#sampD2 = norm.rvs(loc=2.0,scale=0.5,size=300)


import seaborn as sns



def RunAllRTM():
    joined= pd.concat([rs.ExtractBands_colstack('before', 'normed_'), rs.ExtractBands_colstack('after', 'normed_')], axis = 1)
    names, conditions, electrodes = rs.LoadRestingInfo()

    cond_dict = {'plus': 0, 'minus':1, 'sham':2, 'control':3}

    conditions_coded = [cond_dict[label] for label in conditions]

    joined['conditions'] = conditions
    joined['conditions_coded'] = conditions_coded
    joined['names'] = names

    for band in ['alpha', 'smr', 'beta1', 'beta2', 'theta', 'trained']:
        RTM(joined, band)


def RTM(joined, band):
    """regression to the mean"""
#    fig = plt.figure()
#    corr = fig.add_subplot(311)
#    change = fig.add_subplot(312)
#    hist = fig.add_subplot(313)

    global initial

    initial = joined[band +'_before'].as_matrix()

    followUp =  joined[band+'_after'].as_matrix()

    #change.scatter(initial,initial - followUp, color = 'b', alpha = 0.5)
    #corr.scatter(initial, followUp)

    #predicted = NormalModel(initial, followUp)

    #change.scatter(initial,predicted, color ='r', alpha = 0.5)


    GeneralModel(initial, followUp, band, joined['conditions_coded'], joined['conditions'])



def GeneralModel(_initial, followUp, band, conditions, conditions_str):
    path ='/Users/ryszardcetnarski/Desktop/RTM/ModelInfo/'

    global kde_bandwith
    kde_bandwith= 0.2
    #Reshape
    global initial
    initial = _initial *1000
    followUp = followUp *1000
    #Correlation
    corr, p_val = pearsonr(initial, followUp)

    #KDE using scipy, returns a function
    dist_func = gaussian_kde(initial, kde_bandwith)


    #Vector for plotting
    x = np.linspace(min(initial),max(initial), 100)


#Plotting
    fig = plt.figure()
    fig.suptitle(band, fontweight = 'bold')
    model = fig.add_subplot(211)
 #   dx = fig.add_subplot(312)
    kernel = fig.add_subplot(212)
    #Plot scipy kernel estimate
    kernel.plot(x,dist_func(x), 'b')

    kernel.hist(initial, normed  =True, color = 'lightBlue')
    kernel.set_xlabel('Kernel density')
    #Calc and plot derivative, just for visualsation
    slope = []
    for i in x:
        slope.append(derivative(dist_func, i))
    #dx.plot(x,slope)
    #dx.set_xlabel('distribution derivative')

    #Calc the predicted difference accrdoing to Das 1983 model
    change = []
    variance = np.var(initial)
    for _initial in initial:
        predicted = (-(1-corr) * variance) * derivative(Dist_LogFunc,_initial)
        change.append(predicted)

    #Plot Das model results
    model.scatter(initial,change, color = 'r', alpha = 0.5, label = 'rtm prediction')
    #Plot original results
    model.scatter(initial,  initial - followUp , color = 'b', alpha = 0.5, label = 'original data')
    #Plot normal distribution model results
    model.scatter(initial, NormalModel(initial, followUp), color = 'g', alpha =0.5)
    model.legend(loc = 'best')
    model.set_xlabel('initial value')
    model.set_ylabel('change value')

    fig.savefig(path +band+'.png', dpi = 400)


    #Plot differences obtained from comparing real and rtm predicted data
    PlotObsModelDiff(initial, followUp, np.array(change), conditions, conditions_str, band)



def Dist_LogFunc(x):
    dist_func = gaussian_kde(initial, kde_bandwith)
    y = np.log(dist_func(x))
    return y

def PlotObsModelDiff(initial, followUp, predicted, conditions, conditions_str, band):
    path ='/Users/ryszardcetnarski/Desktop/RTM/ModelResults/'
    plt.style.use('ggplot')

    predicted = predicted[:,0]
    #Getting results from a model assuming normal distribution
    predictedNormal = NormalModel(initial, followUp)

    #Making a df with field for evaluating the model
    df = pd.DataFrame({'initial':initial,'followUp': followUp, 'predicted':predicted, 'conditions':conditions, 'conditions_str':conditions_str, 'predictedNormal' : predictedNormal} )
    df['diff_not_predicted'] = df['initial'] - df['followUp'] - df['predicted']
    df['predicted_followUp'] = df['initial'] + df['predicted']
    df['FollowUp_contrast'] =df['followUp'] -  df['predicted_followUp']


    #Plotting
    fg = sns.FacetGrid(data=df, hue='conditions_str',aspect=2)
    fg.map(plt.scatter, 'initial', 'diff_not_predicted').add_legend()
    fg.fig.suptitle(band, fontweight = 'bold')

    fig2 = plt.figure()
    fig2.suptitle(band, fontweight = 'bold')
    ax1 = fig2.add_subplot(121)
    ax2 = fig2.add_subplot(122)

    bp = df[['diff_not_predicted', 'conditions_str']].boxplot(by='conditions_str', ax = ax1)
    bp2 = df[['FollowUp_contrast', 'conditions_str']].boxplot(by='conditions_str', ax = ax2)

    fg.savefig(path + band+'_scatter.png', dpi = 300)
    fig2.savefig(path + band+'_box.png',dpi = 300)

def NormalModel(initial, followUp):
    corr, p_val = pearsonr(initial, followUp)

    change = (1-corr) * (initial - np.mean(initial))

    return change

def TryFits():
    size = 30000

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = scipy.arange(size)
    y =  joined['alpha' +'_before']
    ax.hist(y, bins=range(48), color='w', normed = True)

    dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']

    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) #* size
        ax.plot(pdf_fitted, label=dist_name)
        ax.set_xlim(0,47)
    ax.legend(loc='upper right')





def PlotKernel(band = 'alpha'):

    befores = joined[band+'_before'].as_matrix()
    afters = joined[band+'_after'].as_matrix()

    samp_split = np.vstack((befores, afters)).T
    samp_split = np.delete(samp_split, [1,26,36], 0)
    samp_joined = np.hstack((befores, afters))
    samp_joined =  samp_joined[ np.where(samp_joined <26)]
    x = np.linspace(0,28,1000)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    my_pdf = gaussian_kde(samp_joined)
    for i in range (0,len(samp_split)):

      #  kernel = signal.gaussian(np.mean(samp[i,:]), std = 1)
        #sub = samp[i,:]
        # obtaining the pdf (my_pdf is a function!)
      #  my_pdf = gaussian_kde(sub)
        ax.scatter(samp_split[i,0], i, color = 'blue')
        ax.scatter(samp_split[i,1], i, color = 'red')
        # plotting the result

    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111)
    ax.plot(x, my_pdf(x) *400,'r') # distribution function
      #  kernels.append(kernels)
       # hist(samp,normed=1,alpha=.3) # histogram





#Kde using sklearn, returns object
   # kde = KernelDensity(kernel='tophat', bandwidth = 3).fit(initial[:, np.newaxis])
   # log_dens = kde.score_samples(x[:, np.newaxis])

    #Plot sklearn kernel estimate
   # kernel.plot(x, np.exp(log_dens), 'g')
    #Plot original data histogram


  #followUp = np.random.random_sample(100)
    #followUp= np.random.normal(20,10, 100)

    #followUp = np.random.normal(20,10, 100)#initial + np.random.normal(0,100,100)
    #followUp = np.random.random_sample(100)#initial + np.random.normal(0,100,100)
    #hist.hist(initial)

    #initial = np.random.normal(20,10, 100)
    #initial = np.random.random_sample(100)


    #Add noise to each observation
    #initial = #initial *0.95 + np.random.normal(100,100,100)
    #Make a follow up by adding nosie second time to the same population