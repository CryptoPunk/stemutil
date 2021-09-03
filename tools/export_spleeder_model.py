#!/usr/bin/python3
import tensorflow as tf

from spleeter.separator import create_estimator
from spleeter.model import EstimatorSpecBuilder, InputProviderFactory, model_fn, get_model_function
from spleeter.utils.tensor import pad_and_partition


from tensorflow.signal import hann_window, inverse_stft, stft

def build_waveform_input(waveform_input, params):
    waveform = tf.concat(
                [   
                    tf.zeros((params['frame_length'], params['n_channels'])),
                    waveform_input,
                ],  
                0,  
            )   
    mix_stft = tf.transpose(
                stft(
                    tf.transpose(waveform),
                    params['frame_length'],
                    params['frame_step'],
                    window_fn=lambda frame_length, dtype: (
                        hann_window(frame_length, periodic=True, dtype=dtype)
                    ),  
                    pad_end=True,
                ),  
                perm=[1, 2, 0], 
            )

    mix_spectrogram =  tf.abs(
            pad_and_partition(mix_stft, params['T'])
        )[:, :, : params['F'], :]

    return mix_stft, mix_spectrogram

def waveform_serving_input_receiver(params):
    input_features = { 
        "waveform": tf.compat.v1.placeholder(tf.float32, shape=(None, params["n_channels"]), name="waveform"),
    }
    mix_stft, mix_spec = build_waveform_input(input_features['waveform'], params)
    features = {
        "mix_stft": mix_stft,
        "mix_spectrogram": mix_spec
    }
    return tf.estimator.export.ServingInputReceiver(features, input_features)

def spectrogram_serving_input_receiver(params):
    input_features = { 
        "mix_spectrogram": tf.compat.v1.placeholder(tf.float32, shape=(None, params["T"], params["F"], params["n_channels"]), name="mix_waveform"),
    }
    features = dict(input_features.items())
    return tf.estimator.export.ServingInputReceiver(features, input_features)

def export_keras_from_model(params):
    from spleeter.model import get_model_function
    apply_model = get_model_function('unet.unet')
    k_model_input = tf.keras.Input(shape=(params["T"], params["F"], params["n_channels"]), dtype=tf.float32, name='mix_spectrogram')
    k_model_outputs = apply_model(k_model_input, params['instrument_list'], params['model']['params'])
    k_model = tf.keras.Model(inputs=k_model_input, outputs=k_model_outputs, name='4stem')
    
    for var in k_model.variables:
        value = tf.train.load_variable('../stemutil/models/spleeter/4stems', var.name)
        var.assign(value)

    return k_model

def export_from_estimator(params, model_dir):
    def create_estimator(params, MWF, model_dir):
        """ 
        Initialize tensorflow estimator that will perform separation
    
        Params:
        - params: a dictionary of parameters for building the model
    
        Returns:
            a tensorflow estimator
        """
        # Load model.
        params["MWF"] = MWF 
        # Setup config
        session_config = tf.compat.v1.ConfigProto()
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.7 
        config = tf.estimator.RunConfig(session_config=session_config)
        # Setup estimator
        estimator = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=model_dir, params=params, config=config
        )   
        return estimator
    
    
    
    tf.compat.v1.disable_eager_execution()
    estimator = create_estimator(params, False, model_dir)
    receiver = lambda: waveform_serving_input_receiver(params)
    estimator.export_saved_model("4stems-test", receiver)

import os, os.path
import json

cfg_file = "../stemutil/models/spleeter/4stems.json"
params = None
with open(cfg_file) as fh:
    params = json.load(fh)
params["stft_backend"] = "keras"
model_dir = os.path.join(os.path.dirname(cfg_file),params["model_dir"])

model = export_keras_from_model(params)

model.save('4stems.h5', save_format='h5')
'''
spec_builder = EstimatorSpecBuilder(get_input_dict_placeholders(params), params)
apply_model = get_model_function(params['model']['type'])
input_tensor = spec_builder.spectrogram_feature
outputs = apply_model(input_tensor, params['instrument_list'],params['model']['params'])
'''

