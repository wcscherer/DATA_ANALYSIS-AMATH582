#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 3/11/2020

@author: wscherer13

AMATH 582 - Final Project: Building an Improved Music Classifier
"""

import glob
import scipy.io.wavfile as wv
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC as SVC
from sklearn.naive_bayes import GaussianNB
from scipy import signal
from tabulate import tabulate
import pandas as pd
from matplotlib import pyplot as plt 


def read_song(wav_file,tstart, incr):
    """
    This function reads in a wav file and creates a 5 second snippet of it
    
    """
    
    tend = tstart + incr+1
    
    fs, signal = wv.read(wav_file)
    
    mono = np.shape(signal)[1]
    
    if mono == 2:
        
        sig_mon = (signal[:,0]+signal[:,1])/2
        
    else:
        
        sig_mon = signal[0]
    
    sec_inc = sig_mon[tstart*fs:tend*fs]

    return(fs, sec_inc)
    
   
def spectrogram(song, fs):
    """
    Takes and input signal section and sampling rate and converts it to a 
    spectrogram of fft and fft_shift of the input signal. Also produces the
    frequency space of the signal
    
    """

#    fq, t, spect = signal.spectrogram(song,fs,nperseg=int(fs*0.2))
    fq, t, spect = signal.spectrogram(song,fs)
    s_vec = np.reshape(np.real(spect),(np.shape(spect)[0]*np.shape(spect)[1],))
    spec_shape = np.shape(spect)
    
    s_vec = s_vec-np.average(spec_shape)
    
    return(s_vec, spec_shape)

def build_svec(filepath, nclips):
    """
    builds a matrix of spectrograms where each columns is a spectrogram
    of a 5 second clip of a song. Also returns the sampling rate and wave number
    vector of each spectrogram
    """
    song_mat = []
    
    song_path = os.path.join(str(filepath))
    song_files = glob.glob(song_path+'/*.*')
    i = 1
    while i <= nclips:
        start = np.random.randint(10,100)
        for song in song_files:
        
            fs, wav = read_song(song,start,4)
        
            fft_sig, shape = spectrogram(wav,fs)
            
            song_mat.append(fft_sig)
            
        i += 1
        
    song_mat = np.array(song_mat)
        
    return(song_mat, shape, fs)
        
    
def svd_songs(specmat):
    """
    This function performs the svd on an input spectrogram array 
    without using full matrices and returns U, S, V from the decomposition
    """
    n = np.shape(specmat)[1]
    u, s, v = np.linalg.svd(np.transpose(specmat)/np.sqrt(n-1),full_matrices=False)
    
    sv = np.real(np.matmul(np.diag(s),v))
    
    return(u, sv, v, s)
    
def build_train(list_obs):
    """
    Takes in a list of the dataframes to be combined into the dataset
    
    """
    
    newpd = pd.concat(list_obs)
    newpd.reset_index(drop=True, inplace=True)
    
    return(newpd)
    
def train_models(dframe, nfeat, split):
    """
    Splits the input labeled dataframe into train and test splits,
    then trains a LDA classifier, SVM classifier, and KNN classifier
    
    Returns the success rate of each classifier on the test and train data
    """
    
    X_train, X_test, y_train, y_test = train_test_split(dframe.iloc[:,0:nfeat],
                                            dframe['band'], test_size = split)
    
    #Scale input data for classifiers
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)
    
    #Train K Nearest Neighbors Classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train,y_train)
    knn_train = knn.score(X_train,y_train)
    knn_test  = knn.score(X_test,y_test)

    #Train LInear Discriminant Analysis Classifier
    lda_t = LDA()
    lda_t.fit(X_train,y_train)
    lda_train = lda_t.score(X_train,y_train)
    lda_test = lda_t.score(X_test,y_test)

    
    #Train LInear Support Vector Classifier
    lsvc_t = SVC()
    lsvc_t.fit(X_train,y_train)
    lsvc_train = lsvc_t.score(X_train,y_train)
    lsvc_test  = lsvc_t.score(X_test,y_test)
    
    #Train Naive Bayes Classifier
    nbay_t = GaussianNB()
    nbay_t.fit(X_train,y_train)
    nbay_train = nbay_t.score(X_train,y_train)
    nbay_test  = nbay_t.score(X_test,y_test)
    
    return([knn_train,knn_test,lda_train,lda_test,lsvc_train,lsvc_test,nbay_train,nbay_test])
  
    
def avg_models(dframe, nfeat, split, nruns):
    """
    This fuction reads in a dataframe, passes it through the training function
    nrun times and reports the min, max, and average score on the test and 
    train classification for each classifier algorithm
    
    """
    knn_train = []
    knn_test  = []
    lda_train = []
    lda_test  = []
    svc_train = []
    svc_test  = []
    nbay_train = []
    nbay_test  = []
    
    i = 1
    while i <= int(nruns):
        
        vals = train_models(dframe, nfeat, split)
        knn_train.append(vals[0])
        knn_test.append(vals[1])
        lda_train.append(vals[2])
        lda_test.append(vals[3])
        svc_train.append(vals[4])
        svc_test.append(vals[5])
        nbay_train.append(vals[6])
        nbay_test.append(vals[7])
        
        i += 1
    
    knn_train_dat = [min(knn_train), max(knn_train), 
                     np.average(knn_train),np.std(knn_train)]
    knn_test_dat = [min(knn_test), max(knn_test), 
                     np.average(knn_test),np.std(knn_test)]
    
    lda_train_dat = [min(lda_train), max(lda_train), 
                     np.average(lda_train),np.std(lda_train)]
    lda_test_dat = [min(lda_test), max(lda_test), 
                     np.average(lda_test), np.std(lda_test)]
    
    svc_train_dat = [min(svc_train), max(svc_train), 
                     np.average(svc_train), np.std(svc_train)]
    svc_test_dat = [min(svc_test), max(svc_test), 
                     np.average(svc_test), np.std(svc_test)]
    
    nbay_train_dat = [min(nbay_train), max(nbay_train), 
                     np.average(nbay_train), np.std(nbay_train)]
    nbay_test_dat = [min(nbay_test), max(nbay_test), 
                     np.average(nbay_test), np.std(nbay_test)]
    
    data = [knn_train_dat,knn_test_dat,lda_train_dat,lda_test_dat,svc_train_dat
            ,svc_test_dat,nbay_train_dat,nbay_test_dat]
    
    cols = ['KNN Train', 'KNN Test', 'LDA Train', 'LDA Test', 'SVC Train', 
            'SVC Test', 'GNB Train', 'GNB Test']
    idx = ['Min', 'Max', 'Avg', 'Std']
    
    df = pd.DataFrame(np.transpose(data),columns = cols, index = idx)
    
    print('\nTrained Classifiers with',nfeat,'Features \n')
    print(tabulate(df, headers='keys', showindex = 'always', tablefmt='simple'))
    
    return(df)
    
def plot_results(case_dat, modes, split, nruns,title):
    """
    This function returns the plot of the different trained models
    
    """
    
    cases = []
    for mode in modes:
        
        case = avg_models(case_dat,mode,split,nruns)
        cases.append(case)
    
   
    i = 0
    
    knn_train = []
    knn_test  = []
    lda_train = []
    lda_test  = []
    svc_train = []
    svc_test  = []
    nbay_train = []
    nbay_test = []
    for mode in modes:
        knn_train.append(100*cases[i]['KNN Train']['Avg'])
        knn_test.append( 100*cases[i]['KNN Test']['Avg'])
        lda_train.append(100*cases[i]['LDA Train']['Avg'])
        lda_test.append(100*cases[i]['LDA Test']['Avg'])
        svc_train.append(100*cases[i]['SVC Train']['Avg'])
        svc_test.append(100*cases[i]['SVC Train']['Avg'])
        nbay_train.append(100*cases[i]['GNB Train']['Avg'])
        nbay_test.append(100*cases[i]['GNB Test']['Avg'])
        i += 1

    knn_err = []
    knn_ert  = []
    lda_err = []
    lda_ert  = []
    svc_err = []
    svc_ert  = []
    nbay_err = []
    nbay_ert = []
    j = 0
    for mode in modes:
        knn_err.append(100*cases[j]['KNN Train']['Std'])
        knn_ert.append( 100*cases[j]['KNN Test']['Std'])
        lda_err.append(100*cases[j]['LDA Train']['Std'])
        lda_ert.append(100*cases[j]['LDA Test']['Std'])
        svc_err.append(100*cases[j]['SVC Train']['Std'])
        svc_ert.append(100*cases[j]['SVC Train']['Std'])
        nbay_err.append(100*cases[j]['GNB Train']['Std'])
        nbay_ert.append(100*cases[j]['GNB Test']['Std'])
        j += 1    
    

    fig1 = plt.figure(figsize=(9,5))
    plt.subplot(1,2,1)
    plt.errorbar(modes,knn_train, yerr=knn_err, label = 'KNN')
    plt.errorbar(modes,lda_train, yerr=lda_err, label = 'LDA')
    plt.errorbar(modes,svc_train, yerr=svc_err, label = 'SVC')
    plt.errorbar(modes,nbay_train,yerr=nbay_err,label = 'GNB')
    plt.xlabel('Number of Modes Trained On')
    plt.ylabel('AVG Model Score (%)')
    plt.title('Training Results')
    
        
    plt.subplot(1,2,2)
    plt.errorbar(modes,knn_test, yerr=knn_ert, label = 'KNN')
    plt.errorbar(modes,lda_test, yerr=lda_ert, label = 'LDA')
    plt.errorbar(modes,svc_test, yerr=svc_ert, label = 'SVC')
    plt.errorbar(modes,nbay_test,yerr=nbay_ert,label = 'GNB')
    plt.xlabel('Number of Modes Tested On')
    plt.ylabel('AVG Model Score (%)')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.title('Testing Results')
    plt.tight_layout()
    plt.savefig(title+'.png')
    
    return(fig1, cases)
    
def energy_s(s_mat):
    """
    Returns the energy of each component of the s matrix from SVD decomposition
    
    s is a list of singular values
    
    """
    
    itr = int(np.shape(s_mat)[0])
    sums = sum(s_mat)
    norm_1 = s_mat[0]/sums
    norm_c = []
    norm_i = []
    
    for i in range(itr):
        
        norm_c.append(sum(s_mat[0:i+1])/sums)
        norm_i.append(s_mat[i]/sums)
    return(norm_1, norm_c, norm_i) 
    
def plot_svd(sar,title,labels):
    """
    Plots the principle components of a matrix
    """
    
    
    xrang = range(1,len(sar[0])+1)
    fig = plt.figure(figsize=(6,6))
    plt.subplot(2,1,1)
    plt.plot(xrang,sar[0],'ko',label = str(labels[0]))
    plt.plot(xrang,sar[1],'rx',label = str(labels[1]))
    plt.plot(xrang,sar[2],'g*',label = str(labels[2]))
    try:
        plt.plot(xrang,sar[3],'co',label = str(labels[3]))
        plt.ylabel('Sigma Value')
        plt.legend()
        plt.title(str(title))
    except:
        plt.ylabel('Sigma Value')
        plt.legend()
        plt.title(str(title))
    
    plt.subplot(2,1,2)
    plt.semilogy(xrang,sar[0],'ko',label = str(labels[0]))
    plt.semilogy(xrang,sar[1],'rx',label = str(labels[1]))
    plt.semilogy(xrang,sar[2],'g*',label = str(labels[2]))
    try:
        plt.semilogy(xrang,sar[3],'co',label = str(labels[3]))
        plt.xlabel('Component Mode')
        plt.ylabel('log10 Sigma Value')
    except:   
        plt.xlabel('Component Mode')
        plt.ylabel('log10 Sigma Value')
        
    plt.savefig(str(title)+'.png')
    
    return(fig)
    
def rescale_data(svec_list):
    """
    This function rescales the SV* and V matrices by taking the absolute
    value of each matric and then the log10 of each matrix - elementwise. 
    If a value is -inf, it is replaced by -30 
    """
    svec_rsc = []
    
    for svec in svec_list:
        imt = np.log10(np.abs(svec))
        svec_rsc.append(imt.replace(-np.inf,-30))
        
    return(svec_rsc)
        
    
#%%
# Load in training data sets

# Import police clips 
pol_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/Final/police'
police_mat, sp_p, fs_p = build_svec(pol_path,15)

pol_u, pol_sv, pol_v, pol_s = svd_songs(police_mat)

pol_sv = pd.DataFrame(pol_sv)
pol_v = pd.DataFrame(pol_v)
#Feature Scaling
#pol_sv = pd.DataFrame(np.log10(np.abs(pol_sv))) 
#pol_v = pd.DataFrame(np.log10(np.abs(pol_v)))

pol_sv['band']='Police'; pol_v['band']='Police'


#Import lord of the rings clips
lotr_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/Final/lotr'
lotr_mat, sp_l, fs_l = build_svec(pol_path,15)

lotr_u, lotr_sv, lotr_v, lotr_s = svd_songs(lotr_mat)
lotr_sv = pd.DataFrame(lotr_sv)
lotr_v = pd.DataFrame(lotr_v)

#Feature Scaling
#lotr_sv = pd.DataFrame(np.log10(np.abs(lotr_sv)))
#lotr_v = pd.DataFrame(np.log10(np.abs(lotr_v)))

lotr_sv['band']='LoTR'; lotr_v['band']='LoTR'
#

#Import Otis clips
otis_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/Final/otis'
otis_mat, sp_o, fs_o = build_svec(pol_path,15) 
#
otis_u, otis_sv, otis_v, otis_s = svd_songs(otis_mat)
otis_sv = pd.DataFrame(otis_sv)
otis_v = pd.DataFrame(otis_v)

#Feature Scaling
#otis_sv = pd.DataFrame(np.log10(np.abs(otis_sv))) 
#otis_v = pd.DataFrame(np.log10(np.abs(otis_v)))
otis_sv['band']='Otis'; otis_v['band']='Otis'

#create labeled data sets

#%%
#Case 1 Build Model
case1_dat = build_train([otis_sv.replace(-np.inf,-30),
                         lotr_sv.replace(-np.inf,-30),
                         pol_sv.replace(-np.inf,-30)])

otis1, otisc, otisi = energy_s(otis_s)
ploice1, policec, policei = energy_s(pol_s)
lotr1, lotrc, lotri = energy_s(lotr_s)

svecs1 = [otis_s,pol_s,lotr_s]
labels1 = ['otis','police','lotr']
title1 = 'Case 1 Principle Components'
# Plotting the energies of principle components for each artist
fig1 = plot_svd(svecs1,title1,labels1)


fig2 = plt.figure(figsize=(6,5))
plt.subplot(2,1,1)
plt.plot(range(1,np.shape(otis_s)[0]+1),otisi,'kx', label = 'otis')
plt.plot(range(1,np.shape(pol_s)[0]+1),policei,'ro', label = 'police')
plt.plot(range(1,np.shape(lotr_s)[0]+1),lotri,'g*', label = 'lotr')
plt.ylabel('Component Energy')
plt.title('Case 1 Component Energies')
plt.legend()

plt.subplot(2,1,2)
plt.plot(range(1,np.shape(otis_s)[0]+1),otisc,'kx', label = 'otis')
plt.plot(range(1,np.shape(pol_s)[0]+1),policec,'ro', label = 'police')
plt.plot(range(1,np.shape(lotr_s)[0]+1),lotrc,'g*', label = 'lotr')
plt.ylabel('Cumulative Energy')
plt.xlabel('Component Mode')
plt.savefig('case1_snorm_noscale.png')

#%% train models case 1

fig1, case1 = plot_results(case1_dat, [5,10,25,50, 100,150],0.20,10,'Artists_score_noscale_30')

#%%
#Case 2 Build Band Classifier in Same Genre

# Led Zeppelin Data
led_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/led_zep'
led_mat, kp_l, fs_l = build_svec(led_path,15)

led_u, led_sv, led_v, led_s = svd_songs(led_mat)

led_sv = pd.DataFrame(np.log10(np.abs(led_sv))) 
led_v = pd.DataFrame(np.log10(np.abs(led_v)))

led_sv['band']='Zeppelin'; led_v['band']='Zeppelin'  


# Jimi Hendrix Data
jimi_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/jimi'
jimi_mat, kp_j, fs_j = build_svec(jimi_path,15)

jimi_u, jimi_sv, jimi_v, jimi_s = svd_songs(jimi_mat)

jimi_sv = pd.DataFrame(np.log10(np.abs(jimi_sv))) 
jimi_v = pd.DataFrame(np.log10(np.abs(jimi_v)))

jimi_sv['band']='Hendrix'; jimi_v['band']='Hendrix'  

# Stones Hendrix Data
stones_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/stones'
stones_mat, kp_j, fs_j = build_svec(stones_path,15)

stones_u, stones_sv, stones_v, stones_s = svd_songs(stones_mat)

stones_sv = pd.DataFrame(np.log10(np.abs(stones_sv))) 
stones_v = pd.DataFrame(np.log10(np.abs(stones_v)))

stones_sv['band']='Stones'; stones_v['band']='Stones'  
#

#%%
case2_dat = build_train([led_sv.replace(-np.inf,-30), 
                         jimi_sv.replace(-np.inf,-30)
                         ,stones_sv.replace(-np.inf,-30)])
#Case 2 Build Models
led1, ledc, ledi = energy_s(led_s)
jimi1, jimic, jimii = energy_s(pol_s)
stones1, stonesc, stonesi = energy_s(stones_s)

svecs2 = [led_s,jimi_s,stones_s]
labels2 = ['Led Z','Jimi','Stones']
title2 = 'Case 2 Principle Components'
# Plotting the energies of principle components for each artist
fig1 = plot_svd(svecs2,title2,labels2)


fig2 = plt.figure(figsize=(6,5))
plt.subplot(2,1,1)
plt.plot(range(1,np.shape(led_s)[0]+1),ledi,'kx', label = 'led z')
plt.plot(range(1,np.shape(jimi_s)[0]+1),jimii,'ro', label = 'jimi')
plt.plot(range(1,np.shape(stones_s)[0]+1),stonesi,'g*', label = 'stones')
plt.ylabel('Component Energy')
plt.title('Case 2 Component Energies')
plt.legend()

plt.subplot(2,1,2)
plt.plot(range(1,np.shape(led_s)[0]+1),ledc,'kx', label = 'led z')
plt.plot(range(1,np.shape(jimi_s)[0]+1),jimic,'ro', label = 'jimi')
plt.plot(range(1,np.shape(stones_s)[0]+1),stonesc,'g*', label = 'stones')
plt.ylabel('Cumulative Energy')
plt.xlabel('Component Mode')
plt.savefig('case2_snorm_150.png')

#%%
#Case 2 Train Models

fig2, case2 = plot_results(case2_dat, [5,10,25,50,100,150],0.20,10,'CRock_score_150')

#%%
#Case 3 Import Data
#Rock Data
rock_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/rock'
rock_mat, kp_r, fs_r = build_svec(rock_path,15)

rock_u, rock_sv, rock_v, rock_s = svd_songs(rock_mat)

rock_sv = pd.DataFrame(np.log10(np.abs(rock_sv))) 
rock_v = pd.DataFrame(np.transpose(np.log10(np.abs(rock_v))))

rock_sv['band']='Rock'; rock_v['band']='Rock'  


# Jazz Data
jazz_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/jazz'
jazz_mat, kp_j, fs_j = build_svec(jazz_path,15)

jazz_u, jazz_sv, jazz_v, jazz_s = svd_songs(jazz_mat)

jazz_sv = pd.DataFrame(np.log10(np.abs(jazz_sv))) 
jazz_v = pd.DataFrame(np.transpose(np.log10(np.abs(jazz_v))))

jazz_sv['band']='Jazz'; jazz_v['band']='Jazz'  

# Sound Track Data
soul_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/soul'
soul_mat, kp_t, fs_t = build_svec(soul_path,15)

soul_u, soul_sv, soul_v, soul_s = svd_songs(soul_mat)

soul_sv = pd.DataFrame(np.log10(np.abs(soul_sv)))
soul_v = pd.DataFrame(np.transpose(np.log10(np.abs(soul_v))))

soul_sv['band']='Soul'; soul_v['band']='Soul'  
##

#%%
#Case 3 Build Models

case3_dat = build_train([rock_sv.replace(-np.inf,-30), 
                         jazz_sv.replace(-np.inf,-30), 
                         soul_sv.replace(-np.inf,-30)])

rock1, rockc, rocki = energy_s(rock_s)
jazz1, jazzc, jazzi = energy_s(jazz_s)
soul1, soulc, souli = energy_s(soul_s)

svecs3 = [rock_s,jazz_s,soul_s]
labels3 = ['Rock','Jazz','Soul']
title3 = 'Case 3 Principle Components'
# Plotting the energies of principle components for each artist
fig1 = plot_svd(svecs3,title3,labels3)


fig2 = plt.figure(figsize=(6,5))
plt.subplot(2,1,1)
plt.plot(range(1,np.shape(rock_s)[0]+1),rocki,'kx', label = 'rock')
plt.plot(range(1,np.shape(jazz_s)[0]+1),jazzi,'ro', label = 'jazz')
plt.plot(range(1,np.shape(soul_s)[0]+1),souli,'g*', label = 'soul')
plt.ylabel('Component Energy')
plt.title('Case 3 Component Energies')
plt.legend()

plt.subplot(2,1,2)
plt.plot(range(1,np.shape(rock_s)[0]+1),rockc,'kx', label = 'rock')
plt.plot(range(1,np.shape(jazz_s)[0]+1),jazzc,'ro', label = 'jazz')
plt.plot(range(1,np.shape(soul_s)[0]+1),soulc,'g*', label = 'soul')
plt.ylabel('Cumulative Energy')
plt.xlabel('Component Mode')
plt.savefig('case3_snorm_150.png')



fig3, case3 = plot_results(case3_dat, [5,10,25,50,100,150],0.30,10,'Genre_score_150')
#%%
#Case 4 Rage Against The Machine - 4 Albums

# Rage Against of the Machine
ratm_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/Final/ratm'
ratm_mat, kp_r, fs_r = build_svec(ratm_path,15)

ratm_u, ratm_sv, ratm_v, ratm_s = svd_songs(ratm_mat)

ratm_sv = pd.DataFrame(np.log10(np.abs(ratm_sv))) 
ratm_v = pd.DataFrame(np.transpose(np.log10(np.abs(ratm_v))))

ratm_sv['band']='RATM'; ratm_v['band']='RATM'  

# Evil Empire Album
evil_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/Final/evile'
evil_mat, kp_e, fs_e = build_svec(evil_path,15)

evil_u, evil_sv, evil_v, evil_s = svd_songs(evil_mat)

evil_sv = pd.DataFrame(np.log10(np.abs(evil_sv))) 
evil_v = pd.DataFrame(np.transpose(np.log10(np.abs(evil_v))))

evil_sv['band']='EVILE'; evil_v['band']='EVILE'

# Renegades Album
reneg_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/Final/reneg'
reneg_mat, kp_re, fs_re = build_svec(reneg_path,15)

reneg_u, reneg_sv, reneg_v, reneg_s = svd_songs(reneg_mat)

reneg_sv = pd.DataFrame(np.log10(np.abs(reneg_sv))) 
reneg_v = pd.DataFrame(np.transpose(np.log10(np.abs(reneg_v))))

reneg_sv['band']='RENEG'; reneg_v['band']='RENEG'

# The Battle of Los Angeles
tbola_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/Final/tbola'
tbola_mat, kp_tb, fs_tb = build_svec(tbola_path,15)

tbola_u, tbola_sv, tbola_v, tbola_s = svd_songs(tbola_mat)

tbola_sv = pd.DataFrame(np.log10(np.abs(tbola_sv))) 
tbola_v = pd.DataFrame(np.transpose(np.log10(np.abs(tbola_v))))

tbola_sv['band']='TBOLA'; tbola_v['band']='TBOLA'
#%% RATM Build Models
RATM_dat = build_train([ratm_sv.replace(-np.inf,-30), 
                         evil_sv.replace(-np.inf,-30), 
                         reneg_sv.replace(-np.inf,-30),
                         tbola_sv.replace(-np.inf,-30)])

ratm1, ratmc, ratmi = energy_s(ratm_s)
evil1, evilc, evili = energy_s(evil_s)
reneg1, renegc, renegi = energy_s(reneg_s)
tbola1, tbolac, tbolai = energy_s(tbola_s)

svecs4 = [ratm_s,evil_s,reneg_s, tbola_s]
labels4 = ['RATM','EVIL_E','RENEG','TBOLA']
title4 = 'Rage Against the Machine Principle Components'
# Plotting the energies of principle components for each artist
fig4 = plot_svd(svecs4,title4,labels4)


fig5 = plt.figure(figsize=(6,5))
plt.subplot(2,1,1)
plt.plot(range(1,np.shape(ratm_s)[0]+1),ratmi,'kx', label = 'RATM')
plt.plot(range(1,np.shape(evil_s)[0]+1),evili,'ro', label = 'EVIL_E')
plt.plot(range(1,np.shape(reneg_s)[0]+1),renegi,'g*', label = 'RENEG')
plt.plot(range(1,np.shape(tbola_s)[0]+1),tbolai,'co', label = 'TBOLA')
plt.ylabel('Component Energy')
plt.title('Rage Against the Machine Component Energies')
plt.legend()

plt.subplot(2,1,2)
plt.plot(range(1,np.shape(ratm_s)[0]+1),ratmc,'kx', label = 'RATM')
plt.plot(range(1,np.shape(evil_s)[0]+1),evilc,'ro', label = 'EVIL_E')
plt.plot(range(1,np.shape(reneg_s)[0]+1),renegc,'g*', label = 'RENEG')
plt.plot(range(1,np.shape(tbola_s)[0]+1),tbolac,'co', label = 'TBOLA')
plt.ylabel('Cumulative Energy')
plt.xlabel('Component Mode')
plt.tight_layout()
plt.savefig('RATM_snorm_150.png')


#plt.figure(figsize=(8,12))
fig4, case4 = plot_results(RATM_dat, [5,10,25,50,100,150],0.30,10,'RATM_score_150')

#%% Classify songs within Evil Empire

