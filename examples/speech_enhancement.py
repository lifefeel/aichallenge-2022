import os
import tempfile
import soundfile
from espnet2.bin.enh_inference import SeparateSpeech

from utils.asr_utils import convert_to_16k

filepath = '../sample_data/cam1_02.wav'
filename, file_extension = os.path.splitext(filepath)

out_file = filename + '_enhance' + file_extension

t = tempfile.TemporaryDirectory()
out_path = t.name

input_file = convert_to_16k(filepath, tmp_path=out_path)

mixwav_mc, sr = soundfile.read(input_file)
mixwav_sc = mixwav_mc[:]

separate_speech = {}
# For models downloaded from GoogleDrive, you can use the following script:
print("Load model...")
enh_model_sc = SeparateSpeech(
  train_config="../trained_models/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/config.yaml",
  model_file="../trained_models/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/5epoch.pth",
  # for segment-wise process on long speech
  normalize_segment_scale=False,
  show_progressbar=True,
  ref_channel=4,
  normalize_output_wav=True,
  device="cuda:0",
  segment_size=120,
  hop_size=96
)

print("Run speech enhancement...")
wave = enh_model_sc(mixwav_sc[None, ...], sr)

soundfile.write(out_file, wave[0].squeeze(), sr)
print(f"Enhanced speech: {out_file}")