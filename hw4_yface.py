#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:07:07 2020

@author: wscherer13
"""
import numpy as np
import pywt as wv
import cv2
import os
import glob
from matplotlib import pyplot as plt 


def import_image(filepath):
    """
    This function takes a filepath leading to a file with images
    to be imported.  This function uses the opencv function to read in images
    and convert them to greyscale.
    The images in the file are appended to
    a column vector, which is returned by the function
    """
    
    im_path = os.path.join(str(filepath))
    im_files = glob.glob(im_path)
    im_col = []
    
    for imc in im_files:
        #imc = plt.imread(img, 0) # changed from cv2.imread
        img = cv2.imread(imc, cv2.COLOR_BGR2GRAY)
        imcg = np.array(img)
        try:
            xy = np.shape(imcg) #added
        except:
            continue
        imgr = imcg.reshape((xy[0]*xy[1],)) #added
        imgr = np.array(imgr)
        im_col.append(imgr.transpose())
        
    return(im_col)
    
def import_fimage(filepath):
    """
    This function takes a filepath leading to a file with images
    to be imported. This implementation uses the maplotlib image read, which
    reads generic images better and also converts to greyscale.
    The images in the file are appended to
    a column vector, which is returned by the function. Each column is a 
    reshaped image.
    """
    
    im_path = os.path.join(str(filepath))
    im_files = glob.glob(im_path)
    im_col = []
    
    for image in im_files:
        imc = plt.imread(image,0) # changed from cv2.imread
        imcg = np.array(imc)
        try:
            xy = np.shape(imcg) #added
        except:
            continue
        imgr = imcg.reshape((xy[0]*xy[1],)) #added
        imgr = np.array(imgr)
        im_col.append(imgr.transpose())
        
    return(im_col)
    
    
    
def read_files(filepath,forc):
    
    """
    This function is a wrapper for importing the images using import_image 
    functions from each subfolder.  The image files from each folder are 
    concatenated into one large array of all the image files in the directory.
    This image matrix array is returned by the function. It also converts
    the output to float64 for mathematical manipulation.
    """
    
    dir_all = []
    dir_sub = []
    
    path = os.path.join(str(filepath))
    for dir_name in os.listdir(path):
        filad = str(dir_name)
        sub_path = str(filepath)+filad+'/*.*'
        
        if forc == 'cropped':
            dir_sub = import_image(str(sub_path))
        
        if forc == 'full':
            dir_sub = import_fimage(str(sub_path))
            
      
        dir_sub = np.array(dir_sub)
        
        dir_all.append(dir_sub.transpose())
    
    return(np.array(dir_all))
        
def flatten_npar(np_array):
    """
    This function is necessary to combine an array of numpy arrays into
    one large numpy array
    
    """
    
    itr = len(np_array)
    start = np_array[0]
    
    for i in range(1,itr):
        start = np.hstack((start,np_array[i]))
    
    return(np.array(start))
    
    
def plot_svd(s,title):
    
    xrang = range(1,len(s)+1)
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(xrang,s,'ko')
    plt.ylabel('Sigma Value')
    plt.title(str(title))
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.semilogy(xrang,s,'ko')
    plt.xlabel('Component')
    plt.ylabel('log10 Sigma Value')
    plt.grid()
    plt.savefig(str(title)+'.png')
    
    return(fig)
    
    
    
def energy_snorm(s_mat):
    """
    Returns the energy of each component of the s matrix from SVD decomposition
    
    s is a list of singular values
    
    """
    
    itr = int(np.shape(s_mat)[0])
    
    fnorm = np.linalg.norm(s_mat)
    
    norm_2 = s_mat[0]/fnorm
    norm_c = []
    norm_i = []
    
    for i in range(itr):
        
        norm_c.append(np.linalg.norm(s_mat[0:i+1])/fnorm)
        norm_i.append(np.linalg.norm(s_mat[i])/fnorm)
    return(norm_2, norm_c, norm_i)    
    
    
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
    
def sub_mean(nparray):
    """
    This function subtracts the mean from each column in an np array and returns the
    array that has the mean of each column subtracted
    
    """
    
    ncol = np.shape(nparray)[1]
    newarr = np.zeros(np.shape(nparray))
    
    for i in range(ncol):
        
        newarr[:,i] = nparray[:,i]-np.average(nparray[:,i])
    
    return(newarr)
    
    
def svd_images(imagear):
    """
    This function performs the svd on an input image array without using full 
    matrices and returns U, S, V from the decomposition
    """
    n = np.shape(imagear)[1]
    u, s, v = np.linalg.svd(imagear/np.sqrt(n-1),full_matrices=False)
    
    return(u, s, v)
    
def eig_faces(u_mat, nmode, dim):
    """
    This function takes in the principle component modes of an SVD decomp
    and combines nmodes into one output vector for plotting.  Each column in U
    represents the moddes for a principle component and formats the matrix
    back into the photo dimensions
    
    """
    n = int(nmode)
    nparray = np.zeros(np.size(u_mat[:,0]))
    for i in range(n):
        nparray = nparray + u_mat[:,i]
        
    nparray = np.reshape(nparray,dim)
    return(nparray)
        
    


#%% Part 1 - Yale Faces: Perform SVD analysis on all cropped faces and
    #full faces
    
#Import all images into column vectors for cropped and full images
crpd_pth = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/croppedyale/'
im_crpd = read_files(crpd_pth,'cropped')
crpd_dim = (192,168)

crpd_ttl = flatten_npar(im_crpd)
crpd_avg_ttl = sub_mean(crpd_ttl)

full_pth = '/Users/wscherer13/Documents/MATLAB/AMATH582/HW4/yalefaces/yalefaces/'
im_full = read_files(full_pth,'full')
im_full = np.array(im_full)
full_dim = (243,320)
#
full_ttl = flatten_npar(im_full)
full_avg_ttl = sub_mean(full_ttl)

#%% Part 1 Continued - Performing SVD analysis on cropped image data

uca, sca, vca = svd_images(crpd_avg_ttl)

fig1 = plot_svd(sca,'Cropped Images - SVD on Covarianve Matrix')

eng1a0, eng1ac, eng1ai = energy_s(sca)
a12, ac, ai = energy_snorm(sca)

uc, sc, vc = svd_images(crpd_ttl)

fig1 = plot_svd(sc,'Cropped Images - SVD on Raw Images')

eng10, eng1c, eng1i = energy_s(sc)
n12, nc, ni = energy_snorm(sc)

#%%
# Plotting the energies of the cropped pictures
fig3 = plt.figure(3)
plt.subplot(2,1,1)
plt.plot(range(1,np.shape(sc)[0]+1),eng1i,'kx', label = 'raw image data')
plt.plot(range(1,np.shape(sca)[0]+1),eng1ai,'ro', label = 'covar image data')
plt.ylabel('Inidividual Component Energy')
plt.title('Cropped Image Component Energies')
plt.legend()

plt.subplot(2,1,2)
plt.plot(range(1,np.shape(sc)[0]+1),eng1c,'kx', label = 'raw image data')
plt.plot(range(1,np.shape(sca)[0]+1),eng1ac,'ro', label = 'covar image data')
plt.ylabel('Cumulative Energy')
plt.xlabel('Component Mode')
plt.savefig('cropped_snorm.png')
plt.legend()

#%%
#plot the first eigenfaces

eface_nm = np.reshape(uc[:,0],crpd_dim)
eface_av = np.reshape(uca[:,0],crpd_dim)

fig5 = plt.figure(5)
plt.subplot(1,2,1)
plt.pcolor(-1*np.flipud(eface_nm), cmap='gray')
plt.title('Cropped Image 1st Component U[:,0]')

plt.subplot(1,2,2)
plt.pcolor(np.flipud(eface_av), cmap='gray')
plt.title('Cov Cropped Image 1st Mode U[:,0]')


fig6 = plt.figure(6)
nfaces = [5, 10, 100]
i = 1
for n in nfaces:
    faces = eig_faces(uc,n,crpd_dim)
    plt.subplot(1,len(nfaces),i)
    plt.pcolor(-1*np.flipud(faces),cmap='gray')
    plt.title(str(n)+' Modes')
    i += 1
plt.savefig('Cropped_MM_Eig.png')

#%% Part 1 Continued - Performing SVD analysis on cropped image data

ufa, sfa, vfa = svd_images(full_avg_ttl)

fig1 = plot_svd(sfa,'Full Images - SVD on Covariance Matrix')

eng1a0, eng1ac, eng1ai = energy_s(sfa)
fa12, fac, fai = energy_snorm(sfa)

uf, sf, vf = svd_images(full_ttl)

fig1 = plot_svd(sf,'Full Images - SVD on Raw Images')

engf0, engfc, engfi = energy_s(sf)
f12, fc, fi = energy_snorm(sf)

#%%

# Plotting the energies of the full pictures
fig7 = plt.figure(3)
plt.subplot(2,1,1)
plt.plot(range(1,np.shape(sf)[0]+1),engfi,'kx', label = 'raw image data')
plt.plot(range(1,np.shape(sfa)[0]+1),eng1ai,'ro', label = 'covar image data')
plt.ylabel('Inidividual Component Energy')
plt.title('Full Image Component Energies')
plt.legend()

plt.subplot(2,1,2)
plt.plot(range(1,np.shape(sf)[0]+1),engfc,'kx', label = 'raw image data')
plt.plot(range(1,np.shape(sfa)[0]+1),eng1ac,'ro', label = 'covar image data')
plt.ylabel('Cumulative Energy')
plt.xlabel('Component Mode')
plt.savefig('cropped_snorm.png')
plt.legend()

#%%
eface_fm = np.reshape(uf[:,0],full_dim)
eface_af = np.reshape(ufa[:,0],full_dim)

fig8 = plt.figure(8)
plt.subplot(1,2,1)
plt.pcolor(-1*np.flipud(eface_fm), cmap='gray')
plt.title('First Component Mode without Avg Subtracted')

plt.subplot(1,2,2)
plt.pcolor(np.flipud(eface_af), cmap='gray')
plt.title('First Component Mode with Avg Subtracted')


fig6 = plt.figure(6)
nfaces = [1, 10, 100]
i = 1
for n in nfaces:
    faces = eig_faces(uf,n,full_dim)
    plt.subplot(1,len(nfaces),i)
    plt.pcolor(-1*np.flipud(faces),cmap='gray')
    plt.title(str(n)+' Principle Components')
    i += 1
plt.savefig('full_faces_svd')




