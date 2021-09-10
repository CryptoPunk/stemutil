import keras
import math


n_samples = 33
n_levels = 9
n_coeff = 2**n_levels

coeff_kernels = [17]

calc_conv_padding = lambda width, kernel, stride=1: math.ceil(((stride-1)*width-stride+kernel)/2)

model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(n_samples,n_coeff,2), name="input"))
# (time, wavelet, (real,imag), features)
model.add(keras.layers.Reshape(target_shape=(n_samples, n_coeff, 2, 1)))
model.add(keras.layers.ZeroPadding3D(padding=(0, calc_conv_padding(n_coeff, coeff_kernels[0]), 0)))
model.add(keras.layers.Conv3D(filters=64,kernel_size=(17,coeff_kernels[0],1),strides=(1,1,1), activation='relu'))
model.add(keras.layers.ZeroPadding3D(padding=(0, calc_conv_padding(n_coeff, coeff_kernels[0]), 0)))
model.add(keras.layers.Conv3D(filters=128,kernel_size=(17,coeff_kernels[0],1),strides=(1,1,1),activation='relu'))
model.add(keras.layers.Reshape(target_shape=(n_coeff, 2, 128)))
model.add(keras.layers.Dense(2048, activation="relu"))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.ZeroPadding2D(padding=(calc_conv_padding(n_coeff, 33),1)))
model.add(keras.layers.LocallyConnected2D(4,kernel_size=(33,3),activation='relu'))
'''
# (wavelet, (real,imag), wavelet_features, component_features)
model.add(keras.layers.Reshape(target_shape=(n_coeff, 2, 512, 1)))
model.add(keras.layers.ZeroPadding3D(padding=(calc_conv_padding(n_coeff, 65),1,0)))
model.add(keras.layers.Conv3D(filters=512,kernel_size=(65,3,257),strides=(1,1,1)))
'''
'''
# (batch, time, wavelet, features)
#model.add(keras.layers.ZeroPadding2D(padding=(0, 512+767)))
features=[256,64]
# (batch, wavelet, wavelet_features)
model.add(keras.layers.Reshape(target_shape=(-1, features[0], 1)))
# (batch, wavelet, wavelet_features, instrument_features)
model.add(keras.layers.Conv2D(filters=features[1],kernel_size=128,strides=(1,4),padding="same")) # Causal?
model.add(keras.layers.Reshape(target_shape=(2048, -1)))
model.add(keras.layers.Dense(2048, activation="relu"))
# model.add(keras.layers.MaxPool1D(pool_size=10))
# model.add(keras.layers.Conv1D(filters=2048,kernel_size=15,strides=3))
# (batch, wavelet, features)
'''
'''
model.add(keras.layers.Dense(2048, activation="relu"))
#model.add(tf.keras.layers.Permute
#model.add(keras.layers.Conv2D(filters=256,kernel_size=(3,64)) # Causal?
#model.add(keras.layers.Conv2D(filters=1024,kernel_size=(3,16),strides=1)) # Causal?
#model.add(keras.layers.Conv2D(filters=2048,kernel_size=(2,8),strides=3)) # Causal?
#model.add(keras.layers.MaxPool1D(pool_size=2))
#model.add(keras.layers.Conv2D(filters=2048,kernel_size=(2,128),strides=1)) # Causal?
'''
'''
model.add(keras.layers.Reshape(target_shape=(2048, 1)))
model.add(keras.layers.Conv1D(filters=2048,kernel_size=8,strides=1)) # Causal?
model.add(keras.layers.Conv1D(filters=2048,kernel_size=2,strides=3,padding="same")) # Causal?
model.add(keras.layers.Reshape(target_shape=(2, 2048, 1)))
model.add(keras.layers.Conv2D(filters=256,kernel_size=(1,128), padding="same"))
model.add(keras.layers.Conv2D(filters=256,kernel_size=(4,8),padding="same"))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Reshape(target_shape=(-1, 2048, 256, 1)))
model.add(keras.layers.ConvLSTM2D(filters=64, kernel_size=(1,256), stateful=True))
model.add(keras.layers.Reshape(target_shape=(2048, 64)))
model.add(keras.layers.MaxPooling1D(16,data_format='channels_first'))
'''
model.add(keras.layers.Dense(4, activation="relu", name="output",
                                activity_regularizer=keras.regularizers.l1(l1=0.01)))
#model.add(keras.constraints.UnitNorm(axis=-1))

if __name__ == '__main__':
    model.summary()
else:
    model.compile(optimizer=keras.optimizers.get('adadelta'),
                  loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.Accuracy()])
