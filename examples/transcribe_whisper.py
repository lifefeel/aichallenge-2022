import os
import time
from glob import glob

import soundfile
import whisper

model = whisper.load_model("base")

print('model loaded.')

options = {
    'task': 'transcribe',
    'language': 'Korean'
}

data_root = '../sample_data'

todal_audio_duration = 0.0
start_time = time.time()


input_files = glob(os.path.join(data_root, '*.wav'))
wav_files = sorted(input_files)

for filepath in wav_files:
    audio, sr = soundfile.read(filepath, dtype='float32')
    todal_audio_duration += len(audio) / float(sr)
    result = model.transcribe(audio, **options)

    print(filepath)

    for segment in result['segments']:
        print(f'{segment["start"]} - {segment["end"]} : {segment["text"]}')

decode_time = time.time() - start_time
decode_rtf = f'{decode_time / todal_audio_duration:.4f}'

print(f'Total audio duration : {todal_audio_duration:.2f}')
print(f'Todal decode time : {decode_time:.2f}')
print(f'RTF : {decode_rtf}')