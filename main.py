import time

import ffmpeg
import os
import os.path

import yaml
import soundfile
import tempfile

from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification
)

import torch

from model.mission2_model import SpeechEnhancement, VAD, SpeechRecognition
from utils.asr_utils import convert_to_16k
from utils.utils import save_to_json

MODEL_ROOT = '/root/sogang_asr'
TOKENIZER_PATH = os.path.join(MODEL_ROOT, 'threat_model/baseline-kcelectra-newnew_train/tokenizer')
NLP_MODEL_PATH = os.path.join(MODEL_ROOT, 'threat_model/baseline-kcelectra-newnew_train/epoch-26')

def load_settings(settings_path):
    with open(settings_path) as settings_file:
        settings = yaml.safe_load(settings_file)
    return settings


def ffmpeg_extract_wav(input_path, output_path):
    input_stream = ffmpeg.input(input_path)

    output_wav = ffmpeg.output(input_stream.audio, output_path, acodec='pcm_s16le', ac=1, ar='16k')
    output_wav.overwrite_output().run()


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    milli_sec = sec % 1 * 1000
    return '%02d:%02d:%02d.%03d' % (hour, min, sec, milli_sec)


def vad_post_process(speech_ranges):
    last_start = -1
    last_end = 0

    idx = 0
    dialog_ranges = []
    dialog_min_length = 30

    print('=== chunking ===')
    for i, speech_range in enumerate(speech_ranges):
        start_time = speech_range[0]
        end_time = speech_range[1]

        # print(f'({i}) speech : {start_time} to {end_time}')

        if last_start < 0:
            last_start = start_time
            last_end = end_time

        if start_time - last_end > 5:
            # new chunk
            print(
                f'  chunk({idx}) : {convert(last_start)} - {convert(last_end)} (duration : {last_end - last_start})')

            if last_end - last_start > dialog_min_length:
                dialog_ranges.append((last_start, last_end))

            last_start = start_time
            last_end = end_time

            idx += 1
            pass
        else:
            # concat
            last_end = end_time

    if last_end - last_start > dialog_min_length:
        dialog_ranges.append((last_start, last_end))

    return dialog_ranges


def main(filepaths, params, file_type='video'):
    model_params = params["model"]

    global_start_time = time.time()
    total_audio_duration = 0.0
    time_dict = {}
    wave_paths = []

    time_dict['pre'] = {
        'model_running': 0.0
    }

    t = tempfile.TemporaryDirectory()
    out_path = t.name

    if file_type == 'video':
        #
        # mp4 to wav
        #
        for filepath in filepaths:
            model_start_time = time.time()

            print(f'processing : {filepath}')

            filename, file_extension = os.path.splitext(os.path.basename(filepath))
            wav_path = os.path.join(out_path, f'{filename}.wav')
            ffmpeg_extract_wav(filepath, wav_path)
            wave_paths.append(wav_path)

            model_running_time = time.time() - model_start_time

            time_dict['pre']['model_running'] += model_running_time

    else:
        for filepath in filepaths:
            wave_paths.append(convert_to_16k(filepath, out_path))

    #
    # Load Model: Speech Enhancement
    #
    model_start_time = time.time()
    enh_model = SpeechEnhancement(model_params['speech_enhancement'])

    model_loading_time = time.time() - model_start_time

    time_dict['enh'] = {
        'model_loading': model_loading_time,
        'model_running': 0.0
    }

    #
    # Load Model: VAD
    #
    model_start_time = time.time()

    vad_model = VAD(model_params['vad'])

    model_loading_time = time.time() - model_start_time

    time_dict['vad'] = {
        'model_loading': model_loading_time,
        'model_running': 0.0,
    }

    #
    # Load Model: Speech Recognition
    #
    model_start_time = time.time()

    asr_model = SpeechRecognition(model_params['speech_recognition'])

    model_loading_time = time.time() - model_start_time

    time_dict['asr'] = {
        'model_loading': model_loading_time,
        'model_running': 0.0
    }

    #
    # Load Model: Threat Classifier
    #
    model_start_time = time.time()

    tokenizer = ElectraTokenizer.from_pretrained(TOKENIZER_PATH)
    nlp_model = ElectraForSequenceClassification.from_pretrained(NLP_MODEL_PATH)  # 모델 경로 넣기

    model_loading_time = time.time() - model_start_time

    time_dict['nlp'] = {
        'model_loading': model_loading_time,
        'model_running': 0.0
    }

    for wav_path in wave_paths:
        #
        # Speech Enhancement
        #
        model_start_time = time.time()

        mixwav_mc, sr = soundfile.read(wav_path)
        # mixwav.shape: num_samples, num_channels
        # mixwav_sc = mixwav_mc[:,4]
        mixwav_sc = mixwav_mc[:]

        total_audio_duration += len(mixwav_mc) / float(sr)

        filename, file_extension = os.path.splitext(os.path.basename(wav_path))
        enhance_wav_path = os.path.join(out_path, f'{filename}_enhance.wav')

        wave = enh_model.run(mixwav_sc[None, ...], sr)
        soundfile.write(enhance_wav_path, wave[0].squeeze(), sr)

        model_running_time = time.time() - model_start_time

        time_dict['enh']['model_running'] += model_running_time

        #
        # VAD
        #
        model_start_time = time.time()

        speech_ranges = vad_model.inference(enhance_wav_path)

        save_to_json(speech_ranges, 'vad_infer_result.json')

        dialog_ranges = vad_post_process(speech_ranges)

        print('=== dialog part ===')
        audio_list = []
        for i, dialog_range in enumerate(dialog_ranges):
            start_time = dialog_range[0]
            end_time = dialog_range[1]
            print(f'dialog({i}) : {convert(start_time)} - {convert(end_time)} (duration : {end_time - start_time})')

            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)
            audio, _ = soundfile.read(wav_path, start=start_frame, stop=end_frame, dtype='float32')
            audio_list.append((audio, start_time, end_time))

        model_running_time = time.time() - model_start_time

        time_dict['vad']['model_running'] += model_running_time

        #
        # Speech Recognition
        #
        model_start_time = time.time()

        text_list = []
        for audio in audio_list:
            result = asr_model.transcribe(audio[0])

            trans = []
            for segment in result['segments']:
                print(f'{segment["start"]} - {segment["end"]} : {segment["text"]}')
                trans.append(segment["text"])

            text_list.append(' '.join(trans))

        model_running_time = time.time() - model_start_time

        time_dict['asr']['model_running'] += model_running_time

        #
        # Threat Classifier
        #
        model_start_time = time.time()

        pred_list = []
        for text in text_list:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                logits = nlp_model(**inputs).logits

            predicted_class_id = logits.argmax().item()
            pred = nlp_model.config.id2label[predicted_class_id]
            pred_list.append(pred)

        print(pred_list)

        model_running_time = time.time() - model_start_time

        time_dict['nlp']['model_running'] += model_running_time

    global_running_time = time.time() - global_start_time

    print('\n=== Final results ===')
    out_list = []
    for audio, pred_label in zip(audio_list, pred_list):
        start_time = audio[1]
        end_time = audio[2]
        print(f'{start_time} - {end_time} : {pred_label}')
        out_list.append((start_time, end_time, pred_label))

    save_to_json(out_list, 'mission2_result.json')

    print('\n=== Statistics ===')
    for module, elem in time_dict.items():
        print(f'{module} : {elem}')

    print(f'\ntotal audio duration : {total_audio_duration:.2f}')
    print(f'total running time : {global_running_time:.2f}')
    print(f'RTF : {global_running_time / total_audio_duration:.4f}')
    print('finished')

    # TODO : 인식한 시간값을 받아와 계산하여 최종 시간으로 변경
    # 중간 파일을 저장하지 않고 다음 모델로 전달하는 방법
    # 작년것 EPD 돌려보기


if __name__ == '__main__':
    # filepaths = [
    #     '/root/sogang_asr/data/grand2022/cam1_short.mp4',
    #     # '/root/sogang_asr/data/grand2022/cam1_all_noise_unique_v2.mp4',
    #     # '/root/sogang_asr/data/grand2022/cam1_all_noise_30min_v2.mp4',
    # ]
    # main(filepaths)

    filepaths = [
        '/root/sogang_asr/data/grand2022/speech_noise_mixdown_004.wav'
    ]

    params_path = 'config/params.yaml'
    params = load_settings(params_path)

    main(filepaths, params, file_type='audio')

