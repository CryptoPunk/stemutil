import os.path
import json

import numpy as np
import scipy.io.wavfile as wf

import keras

import dtcwpt

def read_coeff(path):
    #TODO: multichannel
    # Create 1-D wave
    sampleRate,signal = wf.read(path)
    # generate DT-CWPT coefficients
    coeff,padding = dtcwpt.forward(signal[...,0:1],10)

    return coeff,padding

def load_coeff(path):
    data_path = os.path.join(path,'data.npz')
    print("Loading test data from {}".format(path))
    if os.path.exists(data_path):
        print("Stored weights found, Loading...")
        return np.load(data_path)
    
    print("Reading Coefficient Data...")
    print("Mixture...")
    mix,padding = read_coeff(os.path.join(path,'mixture.wav'))
    print("Vocals...")
    vocals,_    = read_coeff(os.path.join(path,'vocals.wav'))
    print("Drums...")
    drums,_     = read_coeff(os.path.join(path,'drums.wav'))
    print("Bass...")
    bass,_      = read_coeff(os.path.join(path,'bass.wav'))
    print("Other...")
    other,_     = read_coeff(os.path.join(path,'other.wav'))
    
    coeffs = np.array((vocals,drums,bass,other))
    
    # Transpose to t, coeff, instrument
    coeffs = np.transpose(coeffs,(1,2,0))

    print("Building Weights...")
    #weights = get_weights(coeffs)
    weights = np_gen_weights(coeffs)
    
    training_data = {'mix': mix, 'padding': padding, 'weights': weights}
    
    print("Saving Training Data!")
    np.savez(data_path, **training_data)
    
    return training_data

class musdb18hq(keras.utils.data_utils.Sequence):
    TRAIN='training'
    VALID='validation'
    
    def __init__(self,path,dataset_name=TRAIN,batch_size=2,timesteps=9):
        self.path = path
        self.batch_size = batch_size
        self.timesteps = timesteps
        with open(os.path.join(path,'info.json')) as fh:
            self.dataset = json.load(fh)[dataset_name]

    def _jagged_idx(self,idx):
        j = idx
        for i in range(len(self.dataset)):
            n_timesteps = self.dataset[i].get('n_timesteps',0)
            if j < n_timesteps:
                return i, j
            j -= n_timesteps
        return j

    def _get_timeseries(self, i, j, data):
        offset = self.timesteps // 2
        #pad = (self.timesteps-(data['mix'].shape[0]+1)%16)%16
        m = np.pad(data['mix'],((offset,offset),(0,0)))
        w = np.pad(data['weights'],((offset,offset),(0,0),(0,0)))
        x = np.nan_to_num(m[j:j+self.timesteps])
        y = np.nan_to_num(w[j+self.timesteps//2])
        return (x, y)

    def __len__(self):
        ret = 0
        for sample in self.dataset:
            ret += sample.get('n_timesteps',0)
        return ret//self.batch_size
        #return sum([sample.get('n_timesteps',0) for sample in self.dataset if 'n_timeteps' in sample])//self.batch_size

    def __getitem__(self,idx):
        #dataset = load_dataset(os.path.join(path,directory))
        batch_x = [] #np.ndarray((batch_size, timesteps)+data['mix'].shape[1:])
        batch_y = [] #np.ndarray((batch_size, timesteps)+data['weights'].shape[1:])
        i = None
        data = None
        for batch in range(self.batch_size):
            print(batch)
            new_i, j = self._jagged_idx(idx*self.batch_size+batch)
            if i != new_i:
                i = new_i
                data_path = os.path.join(self.path,self.dataset[i]['weights'])
                data = np.load(data_path, mmap_mode='r')
                #data = load_coeff(data_path)
            x,y = self._get_timeseries(i, j, data)
            batch_x.append(x)
            batch_y.append(y)
        return np.array(batch_x), np.array(batch_y)

    '''
    def load_dataset(path):
        data = load_coeff(path)
        pad = (16-(data['mix'].shape[0]+1)%16)%16
        w = np.pad(data['weights'],((0,pad),(0,0),(0,0)))
        d = np.pad(data['mix'],((0,pad),(0,0)))
        dataset = keras.preprocessing.timeseries.timeseries_dataset_from_array(
            data=d,
            targets=w,
            sequence_length=9,
            batch_size=2)
    
        return dataset
    '''
#test_dataset = load_dataset('Music_Delta_-_Grunge')

#training_datasets = dataset_generator('/media/NAS/syn/musdb18hq/train')
#test_datasets = dataset_generator('/media/NAS/syn/musdb18hq/test')

if __name__ == '__main__':
    import sys

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'gen':
    path = '/media/NAS/syn/musdb18hq/test'
    for directory in os.listdir(path):
        print("Generating %s" % os.path.join(path,directory))
        load_coeff(os.path.join(path,directory))

    path = '/media/NAS/syn/musdb18hq/train'
    for directory in os.listdir(path):
        print("Generating %s" % os.path.join(path,directory))
        load_coeff(os.path.join(path,directory))
