#!/usr/bin/python3
import tensorflow as tf
import stemutil.spleetutil

def split_waveform(waveform, model, params):
    mix_stft, mix_spectrogram = stemutil.spleetutil.waveform_to_spectrogram(mix_waveform, params)

    instruments = model.predict(mix_spectrogram)

    instruments = stemutil.spleetutil.normalize_instrument_masks(instrument_masks, params)

    time_crop = tf.shape(mix_waveform)[0]

    for instrument, instrument_mask in instruments.items():
        del instruments[instrument]
        instrument = instrument[:-len("_spectrogram")]
        instruments[instrument] = stemutil.spleetutil.mask_waveform(instrument_mask, mix_stft, time_crop, params)

    return instruments
