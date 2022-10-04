import os
import time
from glob import glob

import whisper

from utils.asr_utils import get_duration

model = whisper.load_model("medium")

print('model loaded.')

options = {
    'task': 'transcribe',
    'language': 'Korean'
}

data_root = '/root/sogang_asr/data/grand2022/sample_cam1'

todal_audio_duration = 0.0
start_time = time.time()


input_files = glob(os.path.join(data_root, '*.wav'))
wav_files = sorted(input_files)

# files = ['cam1_02.wav', 'cam1_05.wav', 'cam1_06.wav']

for filepath in wav_files:
    # filepath = os.path.join(data_root, file)
    todal_audio_duration += get_duration(filepath)
    result = model.transcribe(filepath, **options)

    print(filepath)

    for segment in result['segments']:
        print(f'{segment["start"]} - {segment["end"]} : {segment["text"]}')

decode_time = time.time() - start_time
decode_rtf = f'{decode_time / todal_audio_duration:.4f}'

print(f'Total audio duration : {todal_audio_duration:.2f}')
print(f'Todal decode time : {decode_time:.2f}')
print(f'RTF : {decode_rtf}')