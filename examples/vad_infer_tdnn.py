import glob
import os
import shutil
import tempfile

from utils.asr_utils import convert_to_16k, vad_tdnn_ims_speech
from utils.utils import save_to_json

filepaths = list(glob.glob(os.path.join('/root/sogang_asr/data/grand2022/cam1_all_noise_unique_v2_enhance.wav')))
dest_path = 'tmp'

try:
    os.mkdir(dest_path)
except FileExistsError:
    pass


for file in filepaths:
    base_name = os.path.basename(file)
    filename, extension = os.path.splitext(base_name)

    out_path = os.path.join(dest_path, filename)
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass

    t = tempfile.TemporaryDirectory()
    tmp_path = t.name

    input_file = convert_to_16k(file, tmp_path=tmp_path)

    results = vad_tdnn_ims_speech(input_file, out_path, ims_speech_path='ims-speech')

    for result in results:
        start_time, end_time = result

        print(f'speech: {start_time:.2f} to {end_time:.2f}')

    save_to_json(results)

    shutil.rmtree(os.path.join(out_path, 'data'))
    shutil.rmtree(os.path.join(out_path, 'mfcc_hires'))
    shutil.rmtree(os.path.join(out_path, 'segmentation'))