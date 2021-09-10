import numpy as np

def complex_to_angle(C):
    return np.angle(C)

def complex_to_radii(C):
    return np.abs(C)

get_weights = np.vectorize(lambda coeff: coeff[np.newaxis]*np.linalg.pinv(coeff[np.newaxis]).T, signature='(i)->(j)')

def complex_to_polar_2d(coeff):
    # (time, coeff, ..., channel)
    input_shape = coeff.shape
    input_dtype = coeff.dtype
    # (time, coeff, (real,imag), ...,  channel)
    output_shape = input_shape[:2] + (2,) + input_shape[2:]
    output_dtype = input_dtype.type().real.dtype
    
    data = np.ndarray(output_shape,dtype=output_dtype)
        
    np.moveaxis(data,2,0)[0] = complex_to_radii(coeff)
    np.moveaxis(data,2,0)[1] = complex_to_angle(coeff)

    return data

