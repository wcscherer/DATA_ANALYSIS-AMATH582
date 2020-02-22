## AMATH 582 HW 3 - Principle Component Analysis; Wyatt Scherer, Due 2/21
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

def greyscale(inpt_npar):
    #import numpy as np
    """Converts RGB data in a numpy array into grey-scale where
    the numpy array is converted to an array of the minimum size
    for all arrays considered
    
    inpt_npar is a numpy array of pictures
    min_size is an integer of the minimum length to cut the number of pictures
    """
    grey_np = []
    
    length = np.shape(inpt_npar)[3]
    
    
    for i in range(int(length)):
        
        grey_np.append(np.dot(inpt_npar[:,:,:,i],[0.2989, 0.5870, 0.1140]))
        
    return(grey_np)
    
def setzerox(input_npar, xmin, xmax):
    #import numpy as np
    """Sets all pixels along x,y from x=0 to xmin and from xmax to end of the
    x domain in the picture format. 
    
    input_npar is a numpy array of a single image
    xmin is the min column to set to zero
    xmax is the max column to set to zero
    """
    
    end = np.shape(input_npar)[1]
    
    for i in range(int(xmin)):
        
        input_npar[:,i] = 0*input_npar[:,i]
    
    for k in range(int(xmax),int(end)):
        
        input_npar[:,k] = 0*input_npar[:,k]
        
    return(input_npar)
    
    
    
def setzeroy(input_npar, ymin, ymax):
    #import numpy as np
    """Sets all pixels along x,y from y=0 to ymin and from ymax to end of the
    y domain in the picture format. 
    
    input_npar is a numpy array of a single image
    ymin is the min row to set to zero
    ymax is the max row to set to zero
    """
    
    end = np.shape(input_npar)[0]
    
    for i in range(int(ymin)):
        
        input_npar[i] = 0*input_npar[i]
    
    for k in range(int(ymax),int(end)-1):
        
        input_npar[k] = 0*input_npar[k]
        
    return(input_npar)
    
    
    
def filter_imX(input_npar,minval,maxval):
    """Applys the setzero filter to array of image arrays
    
    input_array is the input numpy array list of all the grey image files
    minval is the minimum pixel value to filter up to - either x or y
    maxval is the maximum pixel value to filter through the end - x or y
    xory is a string that says to filter along the x or y dimension
    
    """
    #array = np.copy(input_npar)
    try:
        itr = np.shape(input_npar)[0]
        
        for i in range(itr):
            
            input_npar[i] = setzerox(input_npar[i],minval,maxval)
        
        return(input_npar)
        
    except:
        
        return ('ERROR')
    
    
def filter_imY(input_npar,minval,maxval):
    """Applys the setzero filter to array of image arrays
    
    input_array is the input numpy array list of all the grey image files
    minval is the minimum pixel value to filter up to - either x or y
    maxval is the maximum pixel value to filter through the end - x or y
    xory is a string that says to filter along the x or y dimension
    
    """
    #array = np.copy(input_npar)
    try:
        itr = np.shape(input_npar)[0]
        
        for i in range(itr):
            
            input_npar[i] = setzeroy(input_npar[i],minval,maxval)
        
        return(input_npar)
        
    except:
        
        return ('ERROR')
        
        


def build_X(input_npar, filtmin):
    """ 
    This function finds the max pixel value of an input greyscale filtered image
    and returns the max value and its locations in x, y pixels
    """
    
    itr =np.shape(input_npar)[0]
 
    Xavg = []
    Yavg = []
    
    for i in range(int(itr)):
        
        
        Y, X = np.where(input_npar[i] > filtmin)
        
#        Xavg.append(round(np.average(X)))
#        Yavg.append(round(np.average(Y)))
        
        xavg = np.average(X)
        yavg = np.average(Y)
        
        if (np.isnan(xavg)) or (np.isnan(yavg)):
            Xavg.append(np.trunc((680-1)/2))
            Yavg.append(np.trunc((480-1)/2))
        else:
            
            Xavg.append(round(xavg))
            Yavg.append(round(yavg))
            
        
    
    return(np.transpose(Xavg), np.transpose(Yavg))
    
def energy_s(s_mat):
    """
    Returns the energy of each component of the s matrix from SVD decomposition
    
    s is a list of singular values
    
    """
    
    itr = int(np.shape(s_mat)[0])
    
    fnorm = np.linalg.norm(s_mat)
    
    norm_2 = s_mat[0]/fnorm
    norm_e = []
    
    for i in range(itr):
        
        norm_e.append(np.linalg.norm(s_mat[0:i+1])/fnorm)
    
    return(norm_2, norm_e)
    

#%%
# Test Case 1 - simple harmonic motion along z dimension
# import the first data sets from the three cameras
c11 = loadmat('cam1_1.mat')
c21 = loadmat('cam2_1.mat')
c31 = loadmat('cam3_1.mat')

# camera data stored as rgb matrices for each video frame
c11_data = c11['vidFrames1_1']
c21_data = c21['vidFrames2_1']
c31_data = c31['vidFrames3_1']

dat_length = [c11_data.shape[3],c21_data.shape[3],c31_data.shape[3]]
min_l = min(dat_length)

# convert all rgb images to grescale images for easier analysis
c11_dgy = greyscale(c11_data)
c21_dgy = greyscale(c21_data)
c31_dgy = greyscale(c31_data)

# filter the images to show only pixels of the moving can
c11_filt = filter_imX(c11_dgy,300,450)
c21_filt = filter_imX(c21_dgy,250,350)
c31_filt = filter_imY(c31_dgy,235,330)

# create the X matrix to SVD
X1, Y1 = build_X(c11_filt,254)
X2, Y2 = build_X(c21_filt,252)
X3, Y3 = build_X(c31_filt,246)

X1 = X1 - np.average(X1)
Y1 = Y1 - np.average(Y1)

X2 = X2 - np.average(X2)
Y2 = Y2 - np.average(Y2)

X3 = X3 - np.average(X3)
Y3 = Y3 - np.average(Y3)

X = np.vstack((X1,Y1,X2[12:238],Y2[12:238],X3[:226],Y3[:226]))

# perform singular value decomposition on the Covariance matric Cx
Cx = np.cov(X)
u, s1, v = np.linalg.svd(np.transpose(X)/np.sqrt(min_l-1), full_matrices=False)

sig12 = np.power(s1,2)

fig1 = plt.figure(1)
plt.subplot(3,2,1)
plt.plot(range(int(len(s1))),s1,'kx',ms=4)
plt.ylim((-0.8,100))
plt.xlim((-0.3,6))
plt.text(5,85,' (a)')

plt.subplot(3,2,2)
plt.semilogy(range(int(len(s1))),s1,'kx', ms = 4)
plt.xlim((-0.3,6))
plt.ylim((1,100))
plt.text(5.1,50,' (b)')

plt.subplot(3,1,2)
plt.plot(range(np.shape(v)[0]),v[:,0],'k--',label = 'mode 0')
plt.plot(range(np.shape(v)[0]),v[:,1],'b:',label = 'mode 1')
plt.plot(range(np.shape(v)[0]),v[:,2],'r-.',label = 'mode 2')
plt.ylim((-0.5,2))
plt.xlim((-0.5,5.7))
plt.legend(loc='upper left', fontsize = 'small')
plt.text(5.3,1.6,' (c)')

plt.subplot(3,1,3)
plt.plot(range(np.shape(u)[0]),u[:,0],'k--')
plt.plot(range(np.shape(u)[0]),u[:,1],'b:')
plt.plot(range(np.shape(u)[0]),u[:,2],'r-.')
plt.ylim((-0.3,0.3))
plt.grid()
plt.text(210,0.22,' (d)')
plt.savefig('Case1_everything.png')
plt.close()


fig2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(range(np.shape(Y1)[0]),Y1,'k--',label = 'Y cam1')
plt.plot(range(np.shape(Y2)[0]),Y2,'b:', label = 'Y cam2')
plt.plot(range(np.shape(X3)[0]),X3,'r-.', label = 'X cam3')
plt.title('Raw Centered Can Verticle Location')
plt.xlabel('Camera Frame (time)')
plt.ylim((-200,380))
plt.ylabel('Centered Height (z)')
plt.legend( loc='upper left', fontsize = 'small')

plt.subplot(2,1,2)
plt.plot(range(np.shape(Y1)[0]),Y1,'k--',label = 'Y cam1')
plt.plot(range(np.shape(Y2[12:238])[0]),Y2[12:238],'b:', label = 'Y cam2')
plt.plot(range(np.shape(X3[:226])[0]),X3[:226],'r-.', label = 'X cam3')
plt.title('Synchronized Centered Can Verticle Location')
plt.xlabel('Camera Frame (time)')
plt.ylim((-200,380))
plt.ylabel('Centered Height (z)')
plt.legend( loc='upper left', fontsize = 'small')
plt.subplots_adjust(hspace=0.5)
plt.savefig('Case1_ydat.png')
plt.close()


#%%
# Test Case 2 - harmonic motion along z dimension with noisy data
# import the first data sets from the three cameras

c12 = loadmat('cam1_2.mat')
c22 = loadmat('cam2_2.mat')
c32 = loadmat('cam3_2.mat')

# camera data stored as rgb matrices for each video frame
c12_data = c12['vidFrames1_2']
c22_data = c22['vidFrames2_2']
c32_data = c32['vidFrames3_2']

dat_length = [c12_data.shape[3],c22_data.shape[3],c32_data.shape[3]]
min_l = min(dat_length)

# convert all rgb images to grescale images for easier analysis
c12_dgy = greyscale(c12_data)
c22_dgy = greyscale(c22_data)
c32_dgy = greyscale(c32_data)

# filter the images to show only pixels of the moving can
c12_filt = filter_imX(c12_dgy,300,390)
c22_filt = filter_imX(c22_dgy,275,370)
c32_filt = filter_imY(c32_dgy,220,300)

# create the X matrix to SVD
X12, Y12 = build_X(c12_filt,251)
X22, Y22 = build_X(c22_filt,250)
X32, Y32 = build_X(c32_filt,246)

X12 = X12 - np.average(X12)
Y12 = Y12 - np.average(Y12)

X22 = X22 - np.average(X22)
Y22 = Y22 - np.average(Y22)

X32 = X32 - np.average(X32)
Y32 = Y32 - np.average(Y32)

X = np.vstack((X12,Y12,X22[20:334],Y22[20:334],X32[:min_l],Y32[:min_l]))

# perform singular value decomposition on the Covariance matric Cx
Cx = np.cov(X)
u, s2, v = np.linalg.svd(np.transpose(X)/np.sqrt(min_l-1))

sig2 = np.power(s2,2)

fig1 = plt.figure(1)
plt.subplot(3,2,1)
plt.plot(range(int(len(s2))),s2,'kx',ms=4)
plt.ylim((-0.8,100))
plt.xlim((-0.3,6))
plt.text(5,80,' (a)')

plt.subplot(3,2,2)
plt.semilogy(range(int(len(s2))),s2,'kx', ms = 4)
plt.xlim((-0.3,6))
plt.ylim((1,100))
plt.text(5.1,50,' (b)')

plt.subplot(3,1,2)
plt.plot(range(np.shape(v)[0]),v[:,0],'k--',label = 'mode 0')
plt.plot(range(np.shape(v)[0]),v[:,1],'b:',label = 'mode 1')
plt.plot(range(np.shape(v)[0]),v[:,2],'r-.',label = 'mode 2')
plt.ylim((-1,2))
plt.xlim((-0.5,5.7))
plt.legend(loc='upper left', fontsize = 'small')
plt.text(5.3,1.6,' (c)')

plt.subplot(3,1,3)
plt.plot(range(np.shape(u)[0]),u[:,0],'k--')
plt.plot(range(np.shape(u)[0]),u[:,1],'b:')
plt.plot(range(np.shape(u)[0]),u[:,2],'r-.')
plt.ylim((-0.3,0.3))
plt.grid()
plt.text(305,0.22,' (d)')
plt.savefig('Case2_everything.png')
plt.close()


fig2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(range(np.shape(Y12)[0]),Y12,'k--',label = 'Y cam1')
plt.plot(range(np.shape(Y22)[0]),Y22,'b:', label = 'Y cam2')
plt.plot(range(np.shape(X32)[0]),X32,'r-.', label = 'X cam3')
plt.title('Raw Centered Can Verticle Location')
plt.xlabel('Camera Frame (time)')
plt.ylim((-200,380))
plt.ylabel('Centered Height (z)')
plt.legend( loc='upper left', fontsize = 'small')

plt.subplot(2,1,2)
plt.plot(range(np.shape(Y12)[0]),Y12,'k--',label = 'Y cam1')
plt.plot(range(np.shape(Y22[20:334])[0]),Y22[20:334],'b:', label = 'Y cam2')
plt.plot(range(np.shape(X32[:min_l])[0]),X32[:min_l],'r-.', label = 'X cam3')
plt.title('Synchronized Centered Can Verticle Location')
plt.xlabel('Camera Frame (time)')
plt.ylim((-200,380))
plt.ylabel('Centered Height (z)')
plt.legend( loc='upper left', fontsize = 'small')
plt.subplots_adjust(hspace=0.5)
plt.savefig('Case2_ydat.png')
plt.close()


#%%
# Test Case 3 - harmonic motion along z with horizontal displacement
# import the first data sets from the three cameras

c13 = loadmat('cam1_3.mat')
c23 = loadmat('cam2_3.mat')
c33 = loadmat('cam3_3.mat')

# camera data stored as rgb matrices for each video frame
c13_data = c13['vidFrames1_3']
c23_data = c23['vidFrames2_3']
c33_data = c33['vidFrames3_3']

dat_length = [c13_data.shape[3],c23_data.shape[3],c33_data.shape[3]]
min_l = min(dat_length)

# convert all rgb images to grescale images for easier analysis
c13_dgy = greyscale(c13_data)
c23_dgy = greyscale(c23_data)
c33_dgy = greyscale(c33_data)

# filter the images to show only pixels of the moving can
c13_filt = filter_imX(c13_dgy,270,380)
c23_filt = filter_imX(c23_dgy,220,410)
c33_filt = filter_imY(c33_dgy,180,300)

# create the X matrix to SVD
X13, Y13 = build_X(c13_filt,254)
X23, Y23 = build_X(c23_filt,250)
X33, Y33 = build_X(c33_filt,248)

X13 = X13 - np.average(X13)
Y13 = Y13 - np.average(Y13)

X23 = X23 - np.average(X23)
Y23 = Y23 - np.average(Y23)

X33 = X33 - np.average(X33)
Y33 = Y33 - np.average(Y33)
#
X = np.vstack((X13[:217],Y13[:217],X23[26:243],Y23[26:243],X33[20:],Y33[20:]))
#
## perform singular value decomposition on the Covariance matric Cx
#Cx = np.cov(X)
u, s3, v = np.linalg.svd(np.transpose(X)/np.sqrt(min_l-1))
#

sig3 = np.power(s3,2)

fig1 = plt.figure(1)
plt.subplot(3,2,1)
plt.plot(range(int(len(s3))),s3,'kx',ms=4)
plt.ylim((-0.8,80))
plt.xlim((-0.3,6))
plt.text(5,60,' (a)')

plt.subplot(3,2,2)
plt.semilogy(range(int(len(s3))),s3,'kx', ms = 4)
plt.xlim((-0.3,6))
plt.ylim((1,70))
plt.text(5.1,30,' (b)')

plt.subplot(3,1,2)
plt.plot(range(np.shape(v)[0]),v[:,0],'k--',label = 'mode 0')
plt.plot(range(np.shape(v)[0]),v[:,1],'b:',label = 'mode 1')
plt.plot(range(np.shape(v)[0]),v[:,2],'r-.',label = 'mode 2')
plt.ylim((-1,2))
plt.xlim((-0.5,5.7))
plt.legend(loc='upper left', fontsize = 'small')
plt.text(5.3,1.6,' (c)')

plt.subplot(3,1,3)
plt.plot(range(np.shape(u)[0]),u[:,0],'k--')
plt.plot(range(np.shape(u)[0]),u[:,1],'b:')
plt.plot(range(np.shape(u)[0]),u[:,2],'r-.')
plt.ylim((-0.3,0.3))
plt.grid()
plt.text(212,0.22,' (d)')
plt.savefig('Case3_everything.png')
plt.close()


fig2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(range(np.shape(X13)[0]),X13,'k--',label = 'X cam1')
plt.plot(range(np.shape(X23)[0]),X23,'b:', label = 'X cam2')
plt.plot(range(np.shape(Y33)[0]),Y33,'r-.', label = 'Y cam3')
plt.title('Raw Centered Can Horizontal Location')
plt.xlabel('Camera Frame (time)')
plt.ylim((-100,200))
plt.ylabel('Lateral Location (x)')
plt.legend( loc='upper left', fontsize = 'small')

plt.subplot(2,1,2)
plt.plot(range(np.shape(X13[:217])[0]),X13[:217],'k--',label = 'X cam1')
plt.plot(range(np.shape(X23[26:243])[0]),X23[26:243],'b:', label = 'X cam2')
plt.plot(range(np.shape(Y33[20:])[0]),Y33[20:],'r-.', label = 'Y cam3')
plt.title('Synchronized Centered Can Horizontal Location')
plt.xlabel('Camera Frame (time)')
plt.ylim((-100,200))
plt.ylabel('Lateral Location (x)')
plt.legend( loc='upper left', fontsize = 'small')
plt.subplots_adjust(hspace=0.5)
plt.savefig('Case3_ydat.png')
plt.close()


#%%
# Test Case 4 - harmonic motion along z with circular motion in xy
# import the first data sets from the three cameras

c14 = loadmat('cam1_4.mat')
c24 = loadmat('cam2_4.mat')
c34 = loadmat('cam3_4.mat')

# camera data stored as rgb matrices for each video frame
c14_data = c14['vidFrames1_4']
c24_data = c24['vidFrames2_4']
c34_data = c34['vidFrames3_4']

dat_length = [c14_data.shape[3],c24_data.shape[3],c34_data.shape[3]]
min_l = min(dat_length)
startn1 = 0
startn2 = 40
# convert all rgb images to grescale images for easier analysis
c14_dgy = greyscale(c14_data)
c24_dgy = greyscale(c24_data)
c34_dgy = greyscale(c34_data)

# filter the images to show only pixels of the moving can
c14_filt = filter_imX(c14_dgy,320,460)
c24_filt = filter_imX(c24_dgy,210,410)
c34_filt = filter_imY(c34_dgy,150,360)
c34_filt = filter_imX(c34_filt,200,600)

# create the X matrix to SVD
X14, Y14 = build_X(c14_filt,245)
X24, Y24 = build_X(c24_filt,245)
X34, Y34 = build_X(c34_filt,240)

X14 = X14 - np.average(X14)
Y14 = Y14 - np.average(Y14)

X24 = X24 - np.average(X24)
Y24 = Y24 - np.average(Y24)

X34 = X34 - np.average(X34)
Y34 = Y34 - np.average(Y34)

X = np.vstack((X14[:358],Y14[:358],X24[5:363],Y24[5:363],X34[25:383],Y34[25:383]))
#
## perform singular value decomposition on the Covariance matric Cx
#Cx = np.cov(X)
u, s4, v = np.linalg.svd(np.transpose(X)/np.sqrt(min_l-1))
#

sig4 = np.power(s4,2)

fig1 = plt.figure(1)
plt.subplot(3,2,1)
plt.plot(range(int(len(s4))),s4,'kx',ms=4)
plt.ylim((-0.8,80))
plt.xlim((-0.3,6))
plt.text(5,50,' (a)')

plt.subplot(3,2,2)
plt.semilogy(range(int(len(s4))),s4,'kx', ms = 4)
plt.xlim((-0.3,6))
plt.ylim((1,80))
plt.text(5.1,40,' (b)')

plt.subplot(3,1,2)
plt.plot(range(np.shape(v)[0]),v[:,0],'k--',label = 'mode 0')
plt.plot(range(np.shape(v)[0]),v[:,1],'b:',label = 'mode 1')
plt.plot(range(np.shape(v)[0]),v[:,2],'r-.',label = 'mode 2')
plt.ylim((-1,2))
plt.xlim((-0.5,5.7))
plt.legend(loc='upper left', fontsize = 'small')
plt.text(5.3,1.6,' (c)')

plt.subplot(3,1,3)
plt.plot(range(np.shape(u)[0]),u[:,0],'k--')
plt.plot(range(np.shape(u)[0]),u[:,1],'b:')
plt.plot(range(np.shape(u)[0]),u[:,2],'r-.')
plt.ylim((-0.3,0.3))
plt.grid()
plt.text(350,0.22,' (d)')
plt.savefig('Case4_everything.png')
plt.close()


fig2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(range(np.shape(X14)[0]),X14,'k--',label = 'X cam1')
plt.plot(range(np.shape(X24)[0]),X24,'b:', label = 'X cam2')
plt.plot(range(np.shape(Y34)[0]),Y34,'r-.', label = 'Y cam3')
plt.title('Raw Centered Can Horizontal Location')
plt.xlabel('Camera Frame (time)')
plt.ylim((-100,200))
plt.ylabel('Lateral Location (x)')
plt.legend( loc='upper left', fontsize = 'small')

plt.subplot(2,1,2)
plt.plot(range(np.shape(X14[:358])[0]),X14[:358],'k--',label = 'X cam1')
plt.plot(range(np.shape(X24[5:363])[0]),X24[5:363],'b:', label = 'X cam2')
plt.plot(range(np.shape(Y34[25:383])[0]),Y34[25:383],'r-.', label = 'Y cam3')
plt.title('Synchronized Centered Can Horizontal Location')
plt.xlabel('Camera Frame (time)')
plt.ylim((-100,200))
plt.ylabel('Lateral Location (x)')
plt.legend( loc='upper left', fontsize = 'small')
plt.subplots_adjust(hspace=0.5)
plt.savefig('Case4_ydat.png')
plt.close()

#%%
# Final comparison of the four cases - comparing the energy

s1_n2, s1_ne = energy_s(s1)
s2_n2, s2_ne = energy_s(s2)
s3_n2, s3_ne = energy_s(s3)
s4_n2, s4_ne = energy_s(s4)

fig5 = plt.figure(5)
plt.plot(range(6),np.multiply(100,s1_ne),'kx--',label = 'Case 1')
plt.plot(range(6),np.multiply(100,s2_ne),'bx-.',label = 'Case 2')
plt.plot(range(6),np.multiply(100,s3_ne),'rx:', label = 'Case 3')
plt.plot(range(6),np.multiply(100,s4_ne),'gx-', label = 'Case 4')
plt.legend(loc = 'lower right', fontsize = 'small')
plt.xlabel('Principle Component Mode')
plt.ylabel('Cumulative Energy (percent)')
plt.savefig('comp_adat.png')
plt.close()



