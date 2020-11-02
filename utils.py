import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import re
from numpy import genfromtxt
from scipy import signal
from scipy.interpolate import griddata

theta_band = (4,8)
alpha_band = (8,12)
beta_band = (12,40)


def azim_equidist_projection(x, y, z, scale=1.0):
    """Calculate the stereographic projection.
    Given a unit sphere with radius ``r = 1`` and center at
    The origin. Project the point ``p = (x, y, z)`` from the
    sphere's South pole ``(0, 0, -1)`` on a plane on the sphere's
    North pole ``(0, 0, 1)``.
    ``P' = P * (2r / (r + z))``
    Parameters
    ----------
    x, y, z : float
        Positions of electrodes on a unit sphere
    scale : float
        Scale to change the projection point. Defaults to 1,
        which is on the sphere.
    Returns
    -------
    x, y : float
        Positions of electrodes as projected onto a unit circle.
    """
    mu = 1.0 / (scale + z)
    x = x * mu
    y = y * mu
    return np.asarray(x), np.asarray(y)


def get_stft(eeg_signal):
    Fs = 512.0;  # sampling rate
    y = eeg_signal
#   print(y.shape)
    f, ts, Zxx = signal.stft(y,Fs)
    #Y = (abs(np.mean(Zxx,axis = 1)))
    Y = (abs(Zxx))
    return f,abs(Y)
    
def theta_alpha_beta_averages(f,Y):
    theta_range = theta_band
    alpha_range = alpha_band
    beta_range = beta_band
    theta = Y[(f>theta_range[0]) & (f<=theta_range[1])].mean()
    alpha = Y[(f>alpha_range[0]) & (f<=alpha_range[1])].mean()
    beta = Y[(f>beta_range[0]) & (f<=beta_range[1])].mean()
    return theta, alpha, beta
    
    
def make_steps(samples,frame_duration,overlap):
    '''
    in:
    samples - number of samples in the session
    frame_duration - frame duration in seconds 
    overlap - float fraction of frame to overlap in range (0,1)
    
    out: list of tuple ranges
    '''
    #steps = np.arange(0,len(df),frame_length)
    Fs = 500
    i = 0
    intervals = []
    samples_per_frame = Fs * frame_duration
    while i+samples_per_frame <= samples:
        intervals.append((i,i+samples_per_frame-1))
        i = i + samples_per_frame - int(samples_per_frame*overlap)
    return intervals
    
def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] // nElectrodes
    for c in range(int(n_colors)):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i+1, nSamples), end='\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]


def make_frames(df,frame_duration):
    '''
    in: dataframe or array with all channels, frame duration in seconds
    out: array of theta, alpha, beta averages for each probe for each time step
        shape: (n-frames,m-probes,k-brainwave bands)
    '''
    Fs = 512.0
    frame_length = Fs*frame_duration
    frames = []
    epoch_data_frames = []
    steps = make_steps(len(df),frame_duration,overlap)
    for i,_ in enumerate(steps):
        frame = []
        epoch_data_frame = []
        c = []
        if i == 0:
            continue
        else:
            for channel in df.columns:
                snippet = np.array(df.loc[steps[i][0]:steps[i][1],int(channel)])
                f,Y =  get_stft(snippet)
                theta, alpha, beta = theta_alpha_beta_averages(f,Y)
                frame.append([theta, alpha, beta])
                epoch_data_frame.append(snippet)
            
        frames.append(frame)
        epoch_data_frames.append(epoch_data_frame)
    epoch_data_frames = np.swapaxes(np.array(epoch_data_frames),1,2)
    epoch_data_frames = np.concatenate((epoch_data_frames),axis=0)
    return np.array(frames),epoch_data_frames
    

'''
def make_frames(df,frame_duration):
    '''
    in: dataframe or array with all channels, frame duration in seconds
    out: array of theta, alpha, beta averages for each probe for each time step
        shape: (n-frames,m-probes,k-brainwave bands)
    '''
    Fs = 500.0
    frame_length = Fs*frame_duration
    frames = []
    epoch_data_frames = []
    steps = make_steps(len(df),frame_duration,overlap)
    for i,_ in enumerate(steps):
        frame = []
        epoch_data_frame = []
        #c = []
        if i == 0:
            continue
        else:
            for channel in df.columns:
                snippet = np.array(df.loc[steps[i][0]:steps[i][1],int(channel)])
                f,Y,t =  get_stft(snippet)
                print(snippet.shape,Y.shape)
                theta, alpha, beta = theta_alpha_beta_averages(f,Y)
                #print(theta,alpha,beta)
                frame.append([theta, alpha, beta])
                epo_snip = np.array(np.split(snippet,len(t)-1, ))
                epoch_data_frame.append(epo_snip)
        frames.append(frame)
        epoch_data_frames.append(np.array(epoch_data_frame))
    frames = np.array(frames)
    epoch_data_frames = np.array(epoch_data_frames)
    print(frames.shape,epoch_data_frames.shape)
    frames = np.swapaxes(np.swapaxes(frames,1,3),2,3)
    epoch_data_frames = np.swapaxes(np.array(epoch_data_frames),1,2)
    print(frames.shape,epoch_data_frames.shape)
    frames = np.vstack(frames)
    epoch_data_frames = np.vstack(epoch_data_frames)
    print(frames.shape,epoch_data_frames.shape)
    return frames,epoch_data_frames
 '''   


def make_data_pipeline(file_names,labels,image_size,frame_duration,overlap):
    '''
    IN: 
    file_names - list of strings for each input file (one for each subject)
    labels - list of labels for each
    image_size - int size of output images in form (x, x)
    frame_duration - time length of each frame (seconds)
    overlap - float fraction of frame to overlap in range (0,1)
    
    OUT:
    X: np array of frames (unshuffled)
    y: np array of label for each frame (1 or 0)
    '''
    Fs = 512.0   #sampling rate
    frame_length = Fs * frame_duration
    
    print('Generating training data...')
    
    
    for i, file in enumerate(file_names):
        print ('Processing session: ',file, '. (',i+1,' of ',len(file_names),')')
        data = pd.read_excel(file, header=None)
        df = pd.DataFrame(data)
        
        X_0,X_0_eeg = make_frames(df,frame_duration)
        #steps = np.arange(0,len(df),frame_length)
        X_1 = X_0.reshape(len(X_0),10*3)
        
        images = gen_images(np.array(locs_2d),X_1, image_size, normalize=False)
        images = np.swapaxes(images, 1, 3) 
        print(len(images), ' frames generated with label ', labels[i], '.')
        print('\n')
        if i == 0:
            X = images
            
            X_features = X_1
            y = np.ones(len(images))*labels[0]
        else:
            X = np.concatenate((X,images),axis = 0)
            X_features = np.concatenate((X_features,X_1),axis = 0)
            y = np.concatenate((y,np.ones(len(images))*labels[i]),axis = 0)
        
        
    return X,np.array(y),X_features

'''
def make_data_pipeline(file_names,labels,image_size,frame_duration,overlap):
    '''
    IN: 
    file_names - list of strings for each input file (one for each subject)
    labels - list of labels for each
    image_size - int size of output images in form (x, x)
    frame_duration - time length of each frame (seconds)
    
    OUT:
    X: np array of frames (unshuffled)
    y: np array of label for each frame (1 or 0)
    '''
    
    Fs = 512.0   #sampling rate
    frame_length = Fs * frame_duration
    
    print('Generating training data...')
    
    
    for i, file in enumerate(file_names):
        print ('Processing session: ',file, '. (',i+1,' of ',len(file_names),')')
        data = pd.read_excel(file, header=None)
        df = pd.DataFrame(data)
        
        X_0,X_0_eeg = make_frames(df,frame_duration)
        #steps = np.arange(0,len(df),frame_length)
        X_1 = X_0.reshape(len(X_0),10*3)
        
        images = gen_images(np.array(locs_2d),X_1, image_size, normalize=False)
        images = np.swapaxes(images, 1, 3) 
        print(len(images), ' frames generated with label ', labels[i], '.')
        print('\n')
        if i == 0:
            X = images
            X_eeg = X_0_eeg
            y = np.ones(len(images))*labels[0]
        else:
            X = np.concatenate((X,images),axis = 0)
            y = np.concatenate((y,np.ones(len(images))*labels[i]),axis = 0)
            X_eeg = np.concatenate((X_eeg,X_0_eeg),axis = 0)

        
        
    return X,np.array(y),X_eeg
    '''