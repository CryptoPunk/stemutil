#!/usr/bin/python3
import tensorflow as tf

EPSILON = 1e-10
WINDOW_COMPENSATION_FACTOR = 2.0 / 3.0

def pad_and_partition(tensor: tf.Tensor, segment_len: int) -> tf.Tensor:
    """ 
    Pad and partition a tensor into segment of len `segment_len`
    along the first dimension. The tensor is padded with 0 in order
    to ensure that the first dimension is a multiple of `segment_len`.

    Tensor must be of known fixed rank

    Examples:

        ```python
        >>> tensor = [[1, 2, 3], [4, 5, 6]]
        >>> segment_len = 2
        >>> pad_and_partition(tensor, segment_len)
        [[[1, 2], [4, 5]], [[3, 0], [6, 0]]]
        ````

    Parameters:
        tensor (tensorflow.Tensor):
        segment_len (int):

    Returns:
        tensorflow.Tensor:
    """
    tensor_size = tf.math.floormod(tf.shape(tensor)[0], segment_len)
    pad_size = tf.math.floormod(segment_len - tensor_size, segment_len)
    padded = tf.pad(tensor, [[0, pad_size]] + [[0, 0]] * (len(tensor.shape) - 1)) 
    split = (tf.shape(padded)[0] + segment_len - 1) // segment_len
    return tf.reshape(
        padded, tf.concat([[split, segment_len], tf.shape(padded)[1:]], axis=0)
    )   

def waveform_to_spectrogram(waveform_input, params):
    waveform = tf.concat(
                [   
                    tf.zeros((params['frame_length'], params['n_channels'])),
                    waveform_input,
                ],  
                0,  
            )   
    mix_stft = tf.transpose(
                tf.signal.stft(
                    tf.transpose(waveform),
                    params['frame_length'],
                    params['frame_step'],
                    window_fn=lambda frame_length, dtype: (
                        tf.signal.hann_window(frame_length, periodic=True, dtype=dtype)
                    ),  
                    pad_end=True,
                ),  
                perm=[1, 2, 0], 
            )

    mix_spectrogram =  tf.abs(
            pad_and_partition(mix_stft, params['T'])
        )[:, :, : params['F'], :]

    return mix_stft, mix_spectrogram

def normalize_instrument_masks(instrument_masks, params):

    separation_exponent = params["separation_exponent"]
    output_sum = ( 
        tf.reduce_sum(
            [e ** separation_exponent for e in instrument_masks.values()], axis=0
        ) + EPSILON
    )   

    output = {}
    for instrument in instrument_masks.keys():
        # Compute mask with the model
        output[instrument] = (
            instrument_masks[instrument] ** separation_exponent + (EPSILON / len(instrument_masks))
        ) / output_sum

    return output

def mask_waveform(instrument_mask, mix_stft, time_crop, params):
    # Extend mask;
    extension = params["mask_extension"]
    if extension == "average":
        extension_row = tf.reduce_mean(instrument_mask, axis=2, keepdims=True)
    elif extension == "zeros":
        mask_shape = tf.shape(instrument_mask)
        extension_row = tf.zeros((mask_shape[0], mask_shape[1], 1, mask_shape[-1]))
    else:
        raise ValueError()
    n_extra_row = params["frame_length"] // 2 + 1 - params["F"]
    extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
    instrument_mask = tf.concat([instrument_mask, extension], axis=2)

    # Stack back mask.
    old_shape = tf.shape(instrument_mask)
    new_shape = tf.concat(
        [[old_shape[0] * old_shape[1]], old_shape[2:]], axis=0
    )
    instrument_mask = tf.reshape(instrument_mask, new_shape)

    # Remove padded part (for mask having the same size as STFT);

    instrument_mask = instrument_mask[: tf.shape(mix_stft)[0], ...]

    # Build Masked stfts

    instrument_stft = tf.cast(instrument_mask, dtype=tf.complex64) * mix_stft

    # Inverse and reshape the given STFT

    instrument_waveform = (
        tf.signal.inverse_stft(
            tf.transpose(instrument_stft, perm=[2, 0, 1]),
            params["frame_length"],
            params["frame_step"],
            window_fn=lambda frame_length, dtype: (
                tf.signal.hann_window(frame_length, periodic=True, dtype=dtype)
            ),
        )
        * WINDOW_COMPENSATION_FACTOR
    )

    instrument_waveform = tf.transpose(instrument_waveform)


    instrument_waveform = instrument_waveform[params["frame_length"] : params["frame_length"] + time_crop, :]

    return instrument_waveform
