from typing import Optional
import numpy as np
#import numpy.types as npt
import dtcwt
backend = getattr(dtcwt,dtcwt.backend_name)

colfilter = backend.lowlevel.colfilter
coldfilt = backend.lowlevel.coldfilt
colifilt = backend.lowlevel.colifilt

from dtcwt.coeffs import biort as _biort, qshift as _qshift
from dtcwt.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT

interleave_cols = lambda A,B: np.stack((A,B),axis=-1).reshape(-1,A.shape[1]+B.shape[1])

def pad_input(X,n_levels):
    n_samples = X.shape[0]
    chunk_size = 2**n_levels
    padding = (chunk_size-(n_samples%(chunk_size)))%chunk_size
    fpad = -(-padding // 2)
    bpad = padding // 2
    X = np.vstack((np.zeros((fpad,X.shape[1])), X, np.zeros((bpad,X.shape[1]))))
    return X,padding

def strip_output(X,padding):
    fpad = -(-padding // 2)
    bpad = padding // 2
    X = X[fpad:None if bpad == 0 else -bpad]
    return X

def forward(X: np.ndarray, n_levels: Optional[int]=3) -> np.ndarray:
    if len(X.shape) != 2 or X.shape[1] != 1:
        raise ValueError("DT-CWPT X shape must be (n_samples,1), not {}".format(X.shape))
    h0o, g0o, h1o, g1o = _biort(DEFAULT_BIORT)
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(DEFAULT_QSHIFT)

    if n_levels >= 1:
        X = np.column_stack((colfilter(X, h0o),colfilter(X, h1o)))  
    
    for n in range(1,n_levels):
        X = interleave_cols(coldfilt(X, h0b, h0a), coldfilt(X, h1b, h1a))

    return to_complex(X)

def inverse(C: np.ndarray, n_levels: Optional[int]=None):
    if n_levels is None:
        n_levels = int(np.log2(C.shape[1]))
    X = from_complex(C)
    h0o, g0o, h1o, g1o = _biort(DEFAULT_BIORT)
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(DEFAULT_QSHIFT)
    
    for n in range(1,n_levels):
        X = colifilt(X[:,0::2], g0b, g0a) + colifilt(X[:,1::2], g1b, g1a)
    
    if n_levels >= 1:    
        X = colfilter(X[...,0:1], g0o) + colfilter(X[...,1:2], g1o)

    return X

def get_error(X,n_levels):
    F,padding = forward(X,n_levels)
    R = inverse(F,n_levels,padding)
    
    return np.max(np.abs(X - R))

def to_complex(X):
    return X[::2,:] + 1j*X[1::2,:]

def from_complex(C):
    X = np.zeros((C.shape[0]*2, C.shape[1]), dtype=C.real.dtype)
    X[::2, :] = np.real(C)
    X[1::2, :] = np.imag(C)
    return X

if __name__ == '__main__':
    data = np.ndarray((1000,1))
    data.T[0] = range(1000)
    X = forward(data,1)
    #print(get_error(data),3)
