import keras
import icadb

filepath="train.best.hdf5"

'''
try:
    model = keras.models.load_model(filepath)
except Exception:
'''
import keras_model
model = keras_model.model

import os

path = os.path.join(os.getcwd(),'musdb18hq')

training_data = icadb.db(path,
                   ['vocals','drums','bass','other'],
                   dataset_name = icadb.db.TRAIN,
                   batch_size = 8,
                   timesteps = keras_model.n_samples,
                   n_levels = keras_model.n_levels)

validation_data = icadb.db(path,
                     ['vocals','drums','bass','other'],
                     dataset_name = icadb.db.VALID,
                     batch_size = 8,
                     timesteps = keras_model.n_samples,
                     n_levels = keras_model.n_levels)

print(training_data[0][0][0].shape)

#import sys
#sys.exit(0)

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_freq='epoch', save_best_only=True, mode='auto')

model.fit(
    x=training_data,
    #epochs=150,
    validation_data=validation_data,
    callbacks=[checkpoint])
