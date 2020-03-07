#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:10:30 2020

@author: wscherer13
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

    frq, times, spect = signal.spectrogram(song,fs)
#    frq, times, spect = signal.stft(song,fs,nperseg=2048)
    s_vec = np.reshape(np.real(spect),(np.shape(spect)[0]*np.shape(spect)[1],))
    kp = 0
    
    s_vec = s_vec-np.average(s_vec)
    
    return(s_vec, kp)

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
        
            fs, wav = read_song(song,start,5)
        
            fft_sig, kp = spectrogram(wav,fs)
            song_mat.append(fft_sig)
            
        i += 1
        
    song_mat = np.array(song_mat)
        
    return(song_mat, kp, fs)
        
    
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
    lsvc_t = SVC(decision_function_shape = 'ovr')
    lsvc_t.fit(X_train,y_train)
    lsvc_train = lsvc_t.score(X_train,y_train)
    lsvc_test  = lsvc_t.score(X_test,y_test)

    
    return([knn_train,knn_test,lda_train,lda_test,lsvc_train,lsvc_test])
  
    
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
    
    i = 1
    while i <= int(nruns):
        
        vals = train_models(dframe, nfeat, split)
        knn_train.append(vals[0])
        knn_test.append(vals[1])
        lda_train.append(vals[2])
        lda_test.append(vals[3])
        svc_train.append(vals[4])
        svc_test.append(vals[5])
        
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
    
    data = [knn_train_dat,knn_test_dat,lda_train_dat,lda_test_dat,svc_train_dat
            ,svc_test_dat]
    
    cols = ['KNN Train', 'KNN Test', 'LDA Train', 'LDA Test', 'SVC Train', 
            'SVC Test']
    idx = ['Min', 'Max', 'Avg', 'Std']
    
    df = pd.DataFrame(np.transpose(data),columns = cols, index = idx)
    
    print('\nTrained Classifiers with',nfeat,'Features \n')
    print(tabulate(df, headers='keys', showindex = 'always', tablefmt='simple'))
    
    return(df)
    
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
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(xrang,sar[0],'ko',label = str(labels[0]))
    plt.plot(xrang,sar[1],'rx',label = str(labels[1]))
    plt.plot(xrang,sar[2],'g*',label = str(labels[2]))
    plt.ylabel('Sigma Value')
    plt.legend()
    plt.title(str(title))
    
    plt.subplot(2,1,2)
    plt.semilogy(xrang,sar[0],'ko',label = str(labels[0]))
    plt.semilogy(xrang,sar[1],'rx',label = str(labels[1]))
    plt.semilogy(xrang,sar[2],'g*',label = str(labels[2]))
    plt.xlabel('Component Mode')
    plt.ylabel('log10 Sigma Value')
    plt.savefig(str(title)+'.png')
    
    return(fig)
    
    
#%%
# Load in training data sets

# Import police clips 
pol_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/police'
police_mat, kp_p, fs_p = build_svec(pol_path,50)

pol_u, pol_sv, pol_v, pol_s = svd_songs(police_mat)

pol_sv = pd.DataFrame(pol_sv); pol_v = pd.DataFrame(pol_v)

pol_sv['band']='Police'; pol_v['band']='Police'


#Import lord of the rings clips
lotr_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/lotr'
lotr_mat, kp_l, fs_l = build_svec(pol_path,50)

lotr_u, lotr_sv, lotr_v, lotr_s = svd_songs(lotr_mat)

lotr_sv = pd.DataFrame(lotr_sv); lotr_v = pd.DataFrame(lotr_v)

lotr_sv['band']='LoTR'; lotr_v['band']='LoTR'
#

#Import Otis clips
otis_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/otis'
otis_mat, kp_o, fs_o = build_svec(pol_path,50) 
#
otis_u, otis_sv, otis_v, otis_s = svd_songs(otis_mat)

otis_sv = pd.DataFrame(otis_sv); otis_v = pd.DataFrame(otis_v)
otis_sv['band']='Otis'; otis_v['band']='Otis'

# create labeled data sets



#%%
#Case 1 Build Model
case1_dat = build_train([otis_sv,lotr_sv,pol_sv])

otis1, otisc, otisi = energy_s(otis_s)
ploice1, policec, policei = energy_s(pol_s)
lotr1, lotrc, lotri = energy_s(lotr_s)

svecs1 = [otis_s,pol_s,lotr_s]
labels1 = ['otis','police','lotr']
title1 = 'Case 1 Principle Components'
# Plotting the energies of principle components for each artist
fig1 = plot_svd(svecs1,title1,labels1)


fig2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(range(1,np.shape(otis_s)[0]+1),otisi,'kx', label = 'otis')
plt.plot(range(1,np.shape(pol_s)[0]+1),policei,'ro', label = 'police')
plt.plot(range(1,np.shape(lotr_s)[0]+1),lotri,'g*', label = 'lotr')
plt.ylabel('Inidividual Component Energy')
plt.title('Case 1 Component Energies')
plt.legend()

plt.subplot(2,1,2)
plt.plot(range(1,np.shape(otis_s)[0]+1),otisc,'kx', label = 'otis')
plt.plot(range(1,np.shape(pol_s)[0]+1),policec,'ro', label = 'police')
plt.plot(range(1,np.shape(lotr_s)[0]+1),lotrc,'g*', label = 'lotr')
plt.ylabel('Cumulative Energy')
plt.xlabel('Component Mode')
plt.savefig('case1_snorm.png')
plt.legend()
#%% train models case 1
case1_5 = avg_models(case1_dat,5,0.10,10)

case1_100 = avg_models(case1_dat,100,0.10,10)

case1_250 = avg_models(case1_dat,250,0.10,10)

case1_500 = avg_models(case1_dat,500,0.10,10)

#%%
#Case 2 Build Band Classifier in Same Genre

# Led Zeppelin Data
led_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/led_zep'
led_mat, kp_l, fs_l = build_svec(led_path,50)

led_u, led_sv, led_v, led_s = svd_songs(led_mat)

led_sv = pd.DataFrame(led_sv); led_v = pd.DataFrame(led_v)

led_sv['band']='Zeppelin'; led_v['band']='Zeppelin'  


# Jimi Hendrix Data
jimi_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/jimi'
jimi_mat, kp_j, fs_j = build_svec(jimi_path,50)

jimi_u, jimi_sv, jimi_v, jimi_s = svd_songs(jimi_mat)

jimi_sv = pd.DataFrame(jimi_sv); jimi_v = pd.DataFrame(jimi_v)

jimi_sv['band']='Hendrix'; jimi_v['band']='Hendrix'  

# Stones Hendrix Data
stones_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/stones'
stones_mat, kp_j, fs_j = build_svec(stones_path,50)

stones_u, stones_sv, stones_v, stones_s = svd_songs(stones_mat)

stones_sv = pd.DataFrame(stones_sv); stones_v = pd.DataFrame(stones_v)

stones_sv['band']='Stones'; stones_v['band']='Stones'  
#

#%%
case2_dat = build_train([led_sv, jimi_sv, stones_sv])
#Case 2 Build Models
led1, ledc, ledi = energy_s(led_s)
jimi1, jimic, jimii = energy_s(pol_s)
stones1, stonesc, stonesi = energy_s(stones_s)

svecs2 = [led_s,jimi_s,stones_s]
labels2 = ['Led Z','Jimi','Stones']
title2 = 'Case 2 Principle Components'
# Plotting the energies of principle components for each artist
fig1 = plot_svd(svecs2,title2,labels2)


fig2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(range(1,np.shape(led_s)[0]+1),ledi,'kx', label = 'led z')
plt.plot(range(1,np.shape(jimi_s)[0]+1),jimii,'ro', label = 'jimi')
plt.plot(range(1,np.shape(stones_s)[0]+1),stonesi,'g*', label = 'stones')
plt.ylabel('Inidividual Component Energy')
plt.title('Case 2 Component Energies')
plt.legend()

plt.subplot(2,1,2)
plt.plot(range(1,np.shape(led_s)[0]+1),ledc,'kx', label = 'led z')
plt.plot(range(1,np.shape(jimi_s)[0]+1),jimic,'ro', label = 'jimi')
plt.plot(range(1,np.shape(stones_s)[0]+1),stonesc,'g*', label = 'stones')
plt.ylabel('Cumulative Energy')
plt.xlabel('Component Mode')
plt.savefig('case2_snorm.png')
plt.legend()
#%%
#Case 2 Train Models

case2_5 = avg_models(case2_dat,5,0.10,10)

case2_100 = avg_models(case2_dat,100,0.10,10)

case2_250 = avg_models(case2_dat,250,0.10,10)

case2_500 = avg_models(case2_dat,500,0.10,10)

#%%
#Case 3 Import Data
#Rock Data
rock_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/rock'
rock_mat, kp_r, fs_r = build_svec(rock_path,50)

rock_u, rock_sv, rock_v, rock_s = svd_songs(rock_mat)

rock_sv = pd.DataFrame(rock_sv); rock_v = pd.DataFrame(np.transpose(rock_v))

rock_sv['band']='Rock'; rock_v['band']='Rock'  


# Jazz Data
jazz_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/jazz'
jazz_mat, kp_j, fs_j = build_svec(jazz_path,50)

jazz_u, jazz_sv, jazz_v, jazz_s = svd_songs(jazz_mat)

jazz_sv = pd.DataFrame(jazz_sv); jazz_v = pd.DataFrame(np.transpose(jazz_v))

jazz_sv['band']='Jazz'; jazz_v['band']='Jazz'  

# Sound Track Data
soul_path = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/soul'
soul_mat, kp_t, fs_t = build_svec(soul_path,50)

soul_u, soul_sv, soul_v, soul_s = svd_songs(soul_mat)

soul_sv = pd.DataFrame(soul_sv); soul_v = pd.DataFrame(np.transpose(soul_v))

soul_sv['band']='Soul'; soul_v['band']='Soul'  
##

#%%
#Case 3 Build Models

case3_dat = build_train([rock_sv, jazz_sv, soul_sv])

rock1, rockc, rocki = energy_s(rock_s)
jazz1, jazzc, jazzi = energy_s(jazz_s)
soul1, soulc, souli = energy_s(soul_s)

svecs3 = [rock_s,jazz_s,soul_s]
labels3 = ['Rock','Jazz','Soul']
title3 = 'Case 3 Principle Components'
# Plotting the energies of principle components for each artist
fig1 = plot_svd(svecs3,title3,labels3)


fig2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(range(1,np.shape(rock_s)[0]+1),rocki,'kx', label = 'rock')
plt.plot(range(1,np.shape(jazz_s)[0]+1),jazzi,'ro', label = 'jazz')
plt.plot(range(1,np.shape(soul_s)[0]+1),souli,'g*', label = 'soul')
plt.ylabel('Inidividual Component Energy')
plt.title('Case 3 Component Energies')
plt.legend()

plt.subplot(2,1,2)
plt.plot(range(1,np.shape(rock_s)[0]+1),rockc,'kx', label = 'rock')
plt.plot(range(1,np.shape(jazz_s)[0]+1),jazzc,'ro', label = 'jazz')
plt.plot(range(1,np.shape(soul_s)[0]+1),soulc,'g*', label = 'soul')
plt.ylabel('Cumulative Energy')
plt.xlabel('Component Mode')
plt.savefig('case3_snorm.png')
plt.legend()

case3_5 = avg_models(case3_dat,5,0.10,10)

case3_100 = avg_models(case3_dat,100,0.10,10)

case3_250 = avg_models(case3_dat,250,0.25,10)

case3_500 = avg_models(case3_dat,500,0.25,10)