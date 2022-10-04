import os
import tempfile
import soundfile
from espnet2.bin.enh_inference import SeparateSpeech

from utils.asr_utils import convert_to_16k

# from IPython.display import display, Audio
# mixwav_mc, sr = soundfile.read("./content/M05_440C0213_PED_REAL.wav")

filepath = '/root/sogang_asr/data/grand2022/sample_cam1/cam1_02.wav'

t = tempfile.TemporaryDirectory()
out_path = t.name

input_file = convert_to_16k(filepath, tmp_path=out_path)

mixwav_mc, sr = soundfile.read(input_file)
# mixwav.shape: num_samples, num_channels
# mixwav_sc = mixwav_mc[:,4]
mixwav_sc = mixwav_mc[:]
# display(Audio(mixwav_mc.T, rate=sr))



separate_speech = {}
# For models downloaded from GoogleDrive, you can use the following script:
enh_model_sc = SeparateSpeech(
  train_config="/root/sogang_asr/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/config.yaml",
  model_file="/root/sogang_asr/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/5epoch.pth",
  # for segment-wise process on long speech
  normalize_segment_scale=False,
  show_progressbar=True,
  ref_channel=4,
  normalize_output_wav=True,
  device="cuda:0",
)

wave = enh_model_sc(mixwav_sc[None, ...], sr)

print("Enhanced speech", flush=True)
out_file = '/root/sogang_asr/data/grand2022/sample_cam1/cam1_02_enhance.wav'
# scipy.io.wavfile.write(out_file, sr, wave[0].squeeze())

soundfile.write(out_file, wave[0].squeeze(), sr)