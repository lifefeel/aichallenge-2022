import logging
import subprocess
import time
import os
from typing import List

import editdistance
from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile
import speech_recognition as sr

import espnet


def split_audio(file, dest_path='./'):
    print('\nSplitting...')
    sound_file = AudioSegment.from_wav(file)

    audio_chunks = split_on_silence(sound_file,
                                    # must be silent for at least half a second
                                    min_silence_len=500,
                                    keep_silence=True,

                                    # consider it silent if quieter than -16 dBFS
                                    silence_thresh=-35
                                    )

    start_pos = 0
    outputs = []
    for i, chunk in enumerate(audio_chunks):
        duration = chunk.duration_seconds
        end_pos = round(start_pos + duration, 3)
        out_file = os.path.join(dest_path, "chunk_{:04d}.wav".format(i))
        print(f"exporting : {out_file}, {start_pos} - {end_pos}")
        chunk.export(out_file, format="wav")

        outputs.append((out_file, start_pos, end_pos))
        start_pos = end_pos

    return outputs


def split_audio_ims_speech(file, dest_path, ims_speech_path='ims-speech'):
    file = os.path.abspath(file)
    dest_path = os.path.abspath(dest_path)
    current_dir = os.getcwd()
    os.chdir(ims_speech_path)

    vad_file = './vadR1_new.sh'
    result = subprocess.run([vad_file, file, dest_path], capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)

    os.chdir(current_dir)

    # Reading segmentation Result
    with open(os.path.join(dest_path, "segmentation/output_seg/segments"), 'r') as seg_file:
        sound_file = AudioSegment.from_wav(file)
        outputs = []

        for i, line in enumerate(seg_file):
            if not line:
                break
            words = line.split()

            out_file = os.path.join(dest_path, "chunk_{:04d}.wav".format(i))

            start_pos = float(words[2])
            end_pos = float(words[3])

            print(f"exporting : {out_file}, {start_pos} - {end_pos}")
            seg_sound = sound_file[int(start_pos * 1000):int(end_pos * 1000)]
            seg_sound.export(out_file, format="wav")

            outputs.append((out_file, start_pos, end_pos))

        print(f'segments count : {len(outputs)}')
        if len(outputs) == 0:
            outputs.append((file, 0.0, round(sound_file.duration_seconds, 3)))

    return outputs


def recognize_google(file):
    r = sr.Recognizer()
    test_speech = sr.AudioFile(file)
    with test_speech as source:
        # r.adjust_for_ambient_noise(source)
        audio = r.record(source)
    result = r.recognize_google(audio, language='ko-KR', show_all=True)

    return result


def convert_to_16k(input_file, tmp_path='./tmp', return_duration=False):
    try:
        os.mkdir(tmp_path)
    except FileExistsError:
        pass

    sound_file = AudioSegment.from_wav(input_file)

    sound_file = sound_file.set_channels(1)
    sound_file = sound_file.set_frame_rate(16000)

    input_file = os.path.join(tmp_path, 'audio_16k.wav')

    sound_file.export(input_file, format='wav')

    if return_duration:
        return input_file, round(sound_file.duration_seconds, 3)
    else:
        return input_file


def predict_google(input_file, split_mode=0, tmp_path='./tmp'):
    wav_files = get_files_for_asr(input_file, split_mode, tmp_path)

    #
    # 음성인식 처리
    #
    print('\nSpeech recognizing...')
    outputs = []
    for wav_file, start_pos, end_pos in wav_files:
        result = recognize_google(wav_file)
        transcript = result['alternative'][0]['transcript'] if len(result) > 0 else ''

        try:
            print(f'recognized :{wav_file}, {transcript}')
        except UnicodeEncodeError:
            print(f'recognition failed : {wav_file}')

        time.sleep(0.05)

        outputs.append((transcript, start_pos, end_pos))

    return outputs


def get_files_for_asr(input_file, split_mode=0, dest_path='./tmp'):
    # wav를 16k로 변환
    input_file, duration = convert_to_16k(input_file, dest_path, return_duration=True)
    output_list = []

    if split_mode == 1:
        split_results = split_audio(input_file, dest_path=dest_path)
        output_list = split_results
    elif split_mode == 2:
        split_results = split_audio_ims_speech(input_file, dest_path, ims_speech_path='ims-speech')
        output_list = split_results
    else:
        output_list.append((input_file, 0.0, duration))

    return output_list


def predict_espnet(input_file, recognizer, split_mode=0, tmp_path='./tmp'):
    logger = logging.getLogger('asr_server')
    wav_files = get_files_for_asr(input_file, split_mode, tmp_path)

    #
    # 음성인식 처리
    #
    logger.info('Speech recognizing...')
    outputs = []
    for wav_file, start_pos, end_pos in wav_files:
        speech, rate = soundfile.read(wav_file)
        try:
            nbests = recognizer(speech)
            transcript, *_ = nbests[0]
            transcript = transcript.replace('<sos/eos>', '')
        except espnet.nets.pytorch_backend.transformer.subsampling.TooShortUttError:
            transcript = ''
            logger.warning(f'TooShortUttError : {wav_file}')

        try:
            logger.info(f'recognized : {wav_file}, {transcript}')
        except UnicodeEncodeError:
            transcript = ''
            logger.warning(f'UnicodeEncodeError : {wav_file}')

        outputs.append((transcript, start_pos, end_pos))

    return outputs


def get_duration(file):
    with soundfile.SoundFile(file) as f:
        frames = f.frames
        rate = f.samplerate
        duration = frames / float(rate)
        return duration


def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same
    length.
    Args:
      hypotheses: list of hypotheses
      references: list of references
      use_cer: bool, set True to enable cer
    Returns:
      (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer