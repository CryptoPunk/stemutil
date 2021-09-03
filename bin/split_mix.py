#!/usr/bin/python3
import sys

from spleeter.audio.adapter import AudioAdapter

audio_adapter = None
if audio_adapter is None:
    audio_adapter = AudioAdapter.default()

audio_descriptor = sys.argv[1]

offset=0
duration=600.0
#sample_rate = model_config['sample_rate']
sample_rate = 44100

mix_waveform, _ = audio_adapter.load(
    audio_descriptor,
    offset=offset,
    duration=duration,
    sample_rate=sample_rate
)

import stemutil.models
import stemutil.split

model, params = stemutil.models.load_package_model('4stems')
params["mask_extension"] = "average"

stemutil.

from spleeter.audio import Codec
for instrument, data in out_waveforms.items():
    path = './{}.wav'.format(instrument)
    audio_adapter.save(path, data.numpy(), sample_rate, Codec.WAV, "128k")
