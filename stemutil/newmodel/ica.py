import numpy as np

# Perform Morse-Penrose inverse (pinv), to extract signals for ICA
# http://mlg.eng.cam.ac.uk/zoubin/course04/WellingICA.ps
get_weights = np.vectorize(lambda coeff: coeff[np.newaxis]*np.linalg.pinv(coeff[np.newaxis]).T, signature='(i)->(j)')

def complex_to_angle(C):
    return np.angle(C)

def complex_to_magnitude(C):
    return np.abs(C)

class weights:
    def __init__(self,coefficients):
        self.coeff = coefficients

    @property
    def n_coeff(self):
        return len(self.coeff)

    @property
    def coeff_shape(self):
        # coeff shape:  (time, coeff, channels)
        return self.coeff[0].shape

    @property
    def shape(self):
        # return shape: (time, coeff, components, channels)
        return self.coeff_shape[:-1] + (self.n_coeff, + self.coeff_shape[-1])

    @property
    def dtype(self):
        return self.coeff[0].dtype

    def __len__(self):
        return len(self.coeff[0])

    def __getitem__(self, key):
        start, stop, stride = None, None, 1
        if hasattr(key,'__index__'):
            start = key.__index__()
            stop = start + 1
        elif isinstance(key,(slice)):
            start, stop, stride = key.indices(len(self))
            if stride != 1:
                raise IndexError("stride must be 1")

        if not isinstance(start, (int)):
            raise TypeError("indices must be integers")

        return_shape = (stop-start,) + self.shape[1:]

        data = np.ndarray(return_shape,dtype=self.dtype)
        for c in range(self.n_coeff):
            coeff = self.coeff[c][start:stop:stride]
            for chan in range(self.shape[-1]):
                data[...,c,chan] = coeff[...,chan]
        return data

