import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

def zero_pad(X,pad):
    
    X_pad=np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),"constant",constant_values=(0,0))
    return X_pad


def conv_single_step(A_slice_prev,W,b):
    
    # convolution step (Should be called cross correlation)
    Z=np.sum(W*A_slice_prev)+b      # Element wise multiplicaiton
    
    return Z

def conv_forward(A_prev,W,b,hparameters):
    
    # shape of previous activation (m examples, n_H row number,n_W column number,n_C number of channels)
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    
    # shape of filter (n_H number of rows,n_W number of columns, n_C number of channels,n_C number of filters)
    (f,f,n_C_prev,n_C)=W.shape
    
    # hparameters "pad" and "stride"
    pad=hparameters["pad"]
    stride=hparameters["stride"]
    
    # Pad input activation matrix
    A_pad_prev=zero_pad(A_prev,pad)
    
    # Output activation shape
    
    n_H= int((n_H_prev-f+2*pad)/stride)+1
    n_W= int((n_W_prev-f+2*pad)/stride)+1
        # n_C which is the number for fillters in W
    
    Z=np.zeros((m,n_H,n_W,n_C))
    
    
    for i in range(m):                               # loop over the batch of training examples
        
        a_pad_prev = A_pad_prev[i,:,:,:]              # Select ith training example's padded activation
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    h_start=h*stride
                    h_end=h*stride+f
                    w_start=w*stride
                    w_end=w*stride+f
                   
                    
                    a_slice_prev=a_pad_prev[h_start:h_end,w_start:w_end,:]
                    
                    Z[i,h,w,c]=conv_single_step(a_slice_prev,W[:,:,:,c],b[:,:,:,c])
    
    
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache



def pool_forward(A_prev,hparameters):
    
    # shape of previous activation
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    
    # hyper parameters for pooling layer "f" window size, "stride" of each step
    
    f=hparameters["f"]
    stride=hparameters["stride"]
    
    # output activation shape
    
    n_H=int((n_H_prev-f)/stride)+1
    n_W=int((n_W_prev-f)/stride)+1
    n_C=n_C_prev
    
    # initialize output so we can index into it 
    
    A=np.zeros((m,n_H,n_W,n_C))
    
    for i in range(m):
        a_prev=A_prev[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    h_start=h*stride
                    h_end=h*stride+f
                    w_start=w*stride
                    w_end=w*stride+f
                    
                    # slice of A_prev
                    a_slice_prev=a_prev[h_start:h_end,w_start:w_end,c]
                    
                    # maximum value in each slice
                    A[i,h,w,c]=np.max(a_slice_prev)
    
    cache=(A_prev,hparameters)            
    return A,cache



                    
        
        
        
        
    
    
    