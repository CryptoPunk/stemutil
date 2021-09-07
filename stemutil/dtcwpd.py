import numpy as np
import dtcwt
backend = getattr(dtcwt,dtcwt.backend_name)

colfilter = backend.lowlevel.colfilter
coldfilt = backend.lowlevel.coldfilt
colifilt = backend.lowlevel.colifilt

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT

interleave_cols = lambda A,B: np.stack((A,B),axis=-1).reshape(-1,A.shape[1]+B.shape[1])

def pad_input(X,n_levels):
    chunk_size = 4*2**n_levels
    padding = (chunk_size-(X.shape[0]%(chunk_size)))%chunk_size
    fpad = -(-padding // 2)
    bpad = padding // 2
    X = np.vstack((np.zeros((fpad,X.shape[1])), X, np.zeros((bpad,X.shape[1]))))
    return X,padding

def strip_output(X,padding):
    fpad = -(-padding // 2)
    bpad = padding // 2
    X = X[fpad:None if bpad == 0 else -bpad]
    return X

def forward(X,n_levels=3):
    h0o, g0o, h1o, g1o = _biort(DEFAULT_BIORT)
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(DEFAULT_QSHIFT)

    X,padding = pad_input(X,n_levels)
    
    if n_levels >= 1:
        X = np.column_stack((colfilter(X, h0o),colfilter(X, h1o)))  
    
    for n in range(n_levels):
        X = interleave_cols(coldfilt(X, h0b, h0a), coldfilt(X, h1b, h1a))
    
    return X,padding

def inverse(X, n_levels=3, padding=0):
    h0o, g0o, h1o, g1o = _biort(DEFAULT_BIORT)
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(DEFAULT_QSHIFT)
    
    for n in range(n_levels):
        X = colifilt(X[:,0::2], g0b, g0a) + colifilt(X[:,1::2], g1b, g1a)
    
    if n_levels >= 1:    
        X = colfilter(X[...,0:1], g0o) + colfilter(X[...,1:2], g1o)

    X = strip_output(X,padding)

    return X

def get_error(X,n_levels):
    F,padding = forward(X,n_levels)
    R = inverse(F,n_levels,padding)
    
    return np.max(np.abs(X - R))

print(get_error(np.array(range(100)).reshape((-1,1)),1))
