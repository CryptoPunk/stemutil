import numpy as np
import scipy.io.wavfile as wf
import dtcwpt

class dtcwpt_wav(object):
    def __init__(self,path,n_levels=3,dtype=np.dtype('complex64')):
        self.path = path
        self.n_levels = n_levels
        self.dtype = dtype
        self.real_dtype = dtype.type().real.dtype

        signal_sample_rate,signal = wf.read(path, mmap=True)
        self.signal_sample_rate = signal_sample_rate
        self.signal_channels = signal.shape[1]
        self.signal = signal

    @property
    def shape(self):
        return (self.signal_length, self.sample_rate, self.signal_channels)
    
    @property
    def sample_rate(self):
        return 2**self.n_levels

    @property
    def signal_length(self):
        return len(self.signal)

    @property
    def signal_padding(self):
        lpad = self.sample_rate//2
        rpad = (self.signal_length-self.signal_length%self.sample_rate)%self.sample_rate
        return (lpad, rpad)
    
    def __len__(self):
        return (self.signal_length+sum(self.signal_padding))//self.sample_rate

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

        n_timeseries = stop-start
        n_samples = self.sample_rate*n_timeseries
        (lpad, rpad) = self.signal_padding
        lpad = lpad if (start == 0) else 0
        rpad = rpad if (stop == len(self)) else 0

        signal_start = start*self.sample_rate-self.sample_rate//2+lpad
        signal_stop = stop*self.sample_rate-self.sample_rate//2-rpad
        chunk = self.signal[signal_start:signal_stop]
        chunk = np.pad(chunk.astype(self.real_dtype),((lpad,rpad),(0,0)))
        
        packets = np.ndarray((self.signal_channels,n_timeseries,self.sample_rate),dtype=self.dtype)
        for channel in range(self.signal_channels):
            chunk_channel = chunk.T[channel][np.newaxis].T
            packets[channel] = dtcwpt.forward(chunk_channel,self.n_levels)
        #if not isinstance(key,(slice)):
        #    return packets[:,0]
        return np.transpose(packets,(1,2,0))
        #return packets.T

