import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

masking_value = 0#2**0.5

def preprocess_X_train_data(X):
    # Reshape the X array
    reshape = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    
    # Fit the StandardScaler to the reshaped X array
    X_train_mean = np.nanmean(reshape, axis=0)
    X_train_std = np.nanstd(reshape, axis=0)
    #reshape_std = (reshape - X_train_mean) / X_train_std
    
    reshape_std = StandardScaler().fit_transform(reshape) #do it over the training set for normalizing 

    # Create the mask indicating where NaN values were originally located
    mask_X = np.isnan(X).reshape(X.shape[0], X.shape[1], X.shape[2])
    
    # Replace NaN values with -1e9 in X
    reshape_std[np.isnan(reshape_std)] = masking_value
    
    # Reshape the data back to the original shape for X
    X_processed = reshape_std.reshape(X.shape[0], X.shape[1], X.shape[2])

    return X_processed, X_train_mean, X_train_std

def preprocess_X_other_data(X, X_train_mean, X_train_std):
    # Reshape the X array
    reshape = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    
    # Fit the StandardScaler to the reshaped X array
    #X_train_mean = np.nanmean(reshape, axis=0)
    #X_train_std = np.nanstd(reshape, axis=0)
    reshape_std = (reshape - X_train_mean) / X_train_std
    
    #reshape_std = StandardScaler().fit_transform(reshape) #do it over the training set for normalizing 

    # Create the mask indicating where NaN values were originally located
    mask_X = np.isnan(X).reshape(X.shape[0], X.shape[1], X.shape[2])
    
    # Replace NaN values with -1e9 in X
    reshape_std[np.isnan(reshape_std)] = masking_value
    
    # Reshape the data back to the original shape for X
    X_processed = reshape_std.reshape(X.shape[0], X.shape[1], X.shape[2])

    return X_processed

def preprocess_y_data(y, dataset_type):
    # Reshape the y array

    if dataset_type == '_all_stat':
        reshape = y.reshape(y.shape[0] * y.shape[1], y.shape[2])

        mask_y = np.isnan(y).reshape(y.shape[0], y.shape[1], y.shape[2])

        # Replace NaN values with -1e9 in X
        reshape[np.isnan(reshape)] = masking_value
        
        # Reshape the y array
        y_processed = reshape.reshape(y.shape[0], y.shape[1], y.shape[2])
    else:

    #reshape = y.reshape(y.shape[0] * y.shape[1], 1)

    # Fit the StandardScaler to the y array
    #reshape_std = StandardScaler().fit_transform(reshape)

    # Create the mask indicating where NaN values were originally located
        mask_y = np.isnan(y).reshape(y.shape[0], y.shape[1])

        # Replace NaN values with -1e9 in X
        y[np.isnan(y)] = masking_value
        
        # Reshape the y array
        y_processed = y.reshape(y.shape[0], y.shape[1], 1)
    
    return y_processed, mask_y

