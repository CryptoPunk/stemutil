from typing import List, Sequence, Dict
import collections.abc
import os.path
import json
import numpy as np
import keras

import wavsampler
import ica
import icautil

class db_sample(object):
    @classmethod
    def get_wavfile_info(cls, base_path, path):
        sampler = wavsampler.dtcwpt_wav(os.path.join(base_path,path))
        return {
            'path': path,
            'sample_rate': sampler.signal_sample_rate,
            'n_samples': sampler.signal_length,
            'n_channels': sampler.signal_channels,
        }

    @classmethod
    def build_sample(cls, base_path:str, sample_name:str, mix:str):
        info = {
            'name': sample_name,   
            'mix': cls.get_wavfile_info(base_path, mix),
            'components': {}
        }
        for c, path in components.items():
            info['components'][c] = cls.get_wavfile_info(base_path, path) 
            for prop in ['sample_rate','n_samples','n_channels']:
                if info['mix'][prop] != info['components'][c][prop]:
                    raise ValueError('Sample "{}" has mismatched property {}'.format(sample_name,prop))

        return cls(base_path,info,0,components)

    def __init__(self,base_path,sample_info):
        self.base_path = base_path
        self.info = sample_info

    @property
    def signal_sample_rate(self):
        return self.info['mix']['sample_rate']

    @property
    def signal_sample_length(self):
        return self.info['mix']['n_samples']

    @property
    def n_channels(self):
        return self.info['mix']['n_channels']

    def n_timesteps(self,n_levels):
        sample_ratio = 2**n_levels
        lpad = sample_ratio//2
        rpad = (self.signal_sample_length-self.signal_sample_length%sample_ratio)%sample_ratio
        return (self.signal_sample_length+lpad+rpad)//sample_ratio

    def get_mix(self,n_levels):
        full_path = os.path.join(self.base_path,self.info['mix']['path'])
        return wavsampler.dtcwpt_wav(full_path, n_levels)
    
    def get_weights(self,n_levels,component_order):
        coeffs = []
        for c in component_order:
            full_path = os.path.join(self.base_path,self.info['components'][c]['path'])
            coeffs.append(wavsampler.dtcwpt_wav(full_path,n_levels))
        return ica.weights(coeffs)

class db_dataset(collections.abc.Sequence):
    def __init__(self, parent, dataset: list):
        self.parent = parent
        self.dataset = dataset

    def __getitem__(self,*args,**kwargs):
        sample_info = self.dataset.__getitem__(*args,**kwargs)
        return db_sample(self.parent.base_path,sample_info)

    def __setitem__(self,*args,**kwargs):
        self.dataset.__setitem__(*args,**kwargs)

    def append(self, value: db_sample):
        self.dataset.append(value.info)

    def __len__(self,*args,**kwargs):
        return self.dataset.__len__(*args,**kwargs)

class db(keras.utils.data_utils.Sequence):
    TRAIN='training'
    VALID='validation'
    
    def __init__(self, base_path: str,
                       components: Sequence[str],
                       dataset_name: str = TRAIN,
                       batch_size: int = 2, 
                       timesteps: int = 9,
                       n_levels: int = 3):

        self.base_path      = base_path
        self.dataset_name   = dataset_name
        self.batch_size     = batch_size
        self.timesteps      = timesteps
        self.n_levels       = n_levels
        self.components     = components

        self.info = {}

        if os.path.exists(os.path.join(self.base_path,'info.json')):
            self.reload()

    def reload(self):
        with open(os.path.join(self.base_path,'info.json'),'r') as fh:
            self.info = json.load(fh)
            self.dataset = db_dataset(self,self.info[self.dataset_name])

    def save(self):
        with open(os.path.join(self.base_path,'info.json'),'w') as fh:
            json.dump(self.info, fh)

    def add_sample(self, dataset_name: str, sample_name:str, mix:str, components: Dict[str,str]):
        sample = db_sample.build_sample(self.base_path, sample_name, mix, components)
        if dataset_name not in self.info:
            self.info[dataset_name] = []
        self.info[dataset_name].append(sample.info)
        self.save()

    def _jagged_idx(self,idx):
        j = idx
        for i in range(len(self.dataset)):
            n_timesteps = self.dataset[i].n_timesteps(self.n_levels)
            if j < n_timesteps:
                return i, j
            j -= n_timesteps
        raise IndexError("Index out of range")

    def _get_timeseries(self, width, j, data):
        data_len = len(data)
        
        lpad = width // 2 - j
        lpad = lpad if lpad > 0 else 0
        rpad = 1 + width//2 + j - data_len
        rpad = rpad if rpad > 0 else 0

        idx = j - width // 2
        padded_len = data_len + width

        chunk = data[idx+lpad:min(padded_len,idx+width)-rpad]
        chunk = np.pad(chunk,((lpad,rpad),) + ((0,0),)*(len(data.shape)-1))

        return chunk

    def __len__(self):
        ret = 0
        for sample in self.dataset:
            ret += sample.n_timesteps(self.n_levels)
        return ret//self.batch_size
        #return sum([sample.get('n_timesteps',0) for sample in self.dataset if 'n_timeteps' in sample])//self.batch_size

    def __getitem__(self,idx):
        #dataset = load_dataset(os.path.join(path,directory))
        batch_x = [] #np.ndarray((batch_size, timesteps)+data['mix'].shape[1:])
        batch_y = [] #np.ndarray((batch_size, timesteps)+data['weights'].shape[1:])
        i = None
        mix = None
        weights = None
        for batch in range(self.batch_size):
            new_i, j = self._jagged_idx(idx*self.batch_size+batch)
            if i != new_i:
                i = new_i
                mix = self.dataset[i].get_mix(self.n_levels)
                weights = self.dataset[i].get_weights(self.n_levels,self.components)
            x = self._get_timeseries(self.timesteps, j, mix)
            y = self._get_timeseries(1, j, weights)
            x = icautil.complex_to_polar_2d(x)
            y = icautil.complex_to_polar_2d(y)
            # x.shape == (time, coeff, (radii,angle), channels)
            # y.shape == (coeff, (radii,angle), components,  channels)
            # TODO: Support Channels
            x = x[...,0]
            y = y[0,...,0]
            batch_x.append(x)
            batch_y.append(y)
        #return batch_x, batch_y
        return np.array(batch_x), np.array(batch_y)

if __name__ == '__main__':
    path = os.path.join(os.getcwd(),'musdb18hq')

    db = db(path,
                   ['vocals','drums','bass','other'],
                   dataset_name = db.TRAIN,
                   batch_size = 1,
                   timesteps = 9,
                   n_levels = 3)

    if not os.path.exists(os.path.join(path,'info.json')):
        dataset_name = db.TRAIN
        for sample_name in os.listdir(os.path.join(path, 'train')):
            print("Adding '{}' to '{}' dataset".format(sample_name,dataset_name))
            mix = os.path.join('train',sample_name,'mixture.wav')
            components = {}
            for c in ['vocals','drums','bass','other']:
                components[c] = os.path.join('train',sample_name,'{}.wav'.format(c))
            db.add_sample(dataset_name,
                          sample_name,
                          mix, 
                          components)
    
        dataset_name = db.VALID
        for sample_name in os.listdir(os.path.join(path, 'test')):
            print("Adding '{}' to '{}' dataset".format(sample_name,dataset_name))
            mix = os.path.join('test',sample_name,'mixture.wav')
            components = {}
            for c in ['vocals','drums','bass','other']:
                components[c] = os.path.join('test',sample_name,'{}.wav'.format(c))
    
            db.add_sample(dataset_name,
                          sample_name,
                          mix, 
                          components)
    
