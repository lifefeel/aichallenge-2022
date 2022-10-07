import copy
import ffmpeg
import numpy as np
import os
import os.path
from omegaconf import OmegaConf
import soundfile
import tempfile
import wave

from espnet2.bin.enh_inference import SeparateSpeech

import nemo.collections.asr as nemo_asr
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification
)

import torch
from torch.utils.data import DataLoader

import whisper

from utils.utils import save_to_json


def ffmpeg_extract_wav(input_path, output_path):
    input_stream = ffmpeg.input(input_path)

    output_wav = ffmpeg.output(input_stream.audio, output_path, acodec='pcm_s16le', ac=1, ar='16k')
    output_wav.overwrite_output().run()


class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
        self.signal = signal.astype(np.float32) / 32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1


# class for streaming frame-based VAD
# 1) use reset() method to reset FrameVAD's state
# 2) call transcribe(frame) to do VAD on
#    contiguous signal's frames
# To simplify the flow, we use single threshold to binarize predictions.
class FrameVAD:

    def __init__(self, vad_model,
                 model_definition,
                 threshold=0.5,
                 frame_len=2, frame_overlap=2.5,
                 offset=10):
        '''
        Args:
          threshold: If prob of speech is larger than threshold, classify the segment to be speech.
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.vad_model = vad_model

        self.vocab = list(model_definition['labels'])
        self.vocab.append('_')

        self.sr = model_definition['sample_rate']

        self.data_layer = AudioDataLayer(sample_rate=self.sr)
        self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=self.data_layer.collate_fn)

        self.threshold = threshold
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = model_definition['AudioToMFCCPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        self.buffer = np.zeros(shape=2 * self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()

    def _decode(self, frame, offset=0):
        assert len(frame) == self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = self.infer_signal(self.buffer).cpu().numpy()[0]
        decoded = self._greedy_decoder(
            self.threshold,
            logits,
            self.vocab
        )
        return decoded

    @torch.no_grad()
    def transcribe(self, frame=None):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        return unmerged

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer = np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    @staticmethod
    def _greedy_decoder(threshold, logits, vocab):
        s = []
        if logits.shape[0]:
            probs = torch.softmax(torch.as_tensor(logits), dim=-1)
            probas, _ = torch.max(probs, dim=-1)
            probas_s = probs[1].item()
            preds = 1 if probas_s >= threshold else 0
            s = [preds, str(vocab[preds]), probs[0].item(), probs[1].item(), str(logits)]
        return s

    def infer_signal(self, signal):
        self.data_layer.set_signal(signal)
        batch = next(iter(self.data_loader))
        audio_signal, audio_signal_len = batch
        audio_signal, audio_signal_len = audio_signal.to(self.vad_model.device), audio_signal_len.to(self.vad_model.device)
        logits = self.vad_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        return logits


def offline_inference(wave_file, vad_model, model_definition, STEP=0.025, WINDOW_SIZE=0.5, threshold=0.5):
    FRAME_LEN = STEP  # infer every STEP seconds
    CHANNELS = 1  # number of audio channels (expect mono signal)
    RATE = 16000  # sample rate, Hz

    CHUNK_SIZE = int(FRAME_LEN * RATE)

    vad = FrameVAD(vad_model=vad_model,
        model_definition=model_definition,
        threshold=threshold,
        frame_len=FRAME_LEN, frame_overlap=(WINDOW_SIZE - FRAME_LEN) / 2,
        offset=0)

    wf = wave.open(wave_file, 'rb')

    empty_counter = 0

    preds = []
    proba_b = []
    proba_s = []

    data = wf.readframes(CHUNK_SIZE)

    while len(data) > 0:

        data = wf.readframes(CHUNK_SIZE)
        signal = np.frombuffer(data, dtype=np.int16)
        result = vad.transcribe(signal)

        preds.append(result[0])
        proba_b.append(result[2])
        proba_s.append(result[3])

        if len(result):
            # print(result, end='\n')
            empty_counter = 3
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                print(' ', end='')

    # p.terminate()
    vad.reset()

    return preds, proba_b, proba_s


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    milli_sec = sec % 1 * 1000
    return '%02d:%02d:%02d.%03d' % (hour, min, sec, milli_sec)


def main(filepaths):

    for filepath in filepaths:
        #
        # mp4 to wav
        #
        print(f'processing : {filepath}')
        t = tempfile.TemporaryDirectory()
        out_path = t.name

        filename, file_extension = os.path.splitext(os.path.basename(filepath))
        wav_path = os.path.join(out_path, f'{filename}.wav')
        ffmpeg_extract_wav(filepath, wav_path)

        #
        # Speech Enhancement
        #
        mixwav_mc, sr = soundfile.read(wav_path)
        # mixwav.shape: num_samples, num_channels
        # mixwav_sc = mixwav_mc[:,4]
        mixwav_sc = mixwav_mc[:]

        enh_model_sc = SeparateSpeech(
            train_config="/root/sogang_asr/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/config.yaml",
            model_file="/root/sogang_asr/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/5epoch.pth",
            # for segment-wise process on long speech
            normalize_segment_scale=False,
            show_progressbar=True,
            ref_channel=4,
            normalize_output_wav=True,
            device="cuda:0",
            segment_size=120,
            hop_size=96
        )

        enhance_wav_path = os.path.join(out_path, f'{filename}_enhance.wav')

        wave = enh_model_sc(mixwav_sc[None, ...], sr)
        soundfile.write(enhance_wav_path, wave[0].squeeze(), sr)

        #
        # VAD
        #
        vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained('vad_marblenet')
        # Preserve a copy of the full config
        cfg = copy.deepcopy(vad_model._cfg)
        print(OmegaConf.to_yaml(cfg))

        vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)
        # Set model to inference mode
        vad_model.eval()
        vad_model = vad_model.to(vad_model.device)

        model_definition = {
            'sample_rate': cfg.train_ds.sample_rate,
            'AudioToMFCCPreprocessor': cfg.preprocessor,
            'JasperEncoder': cfg.encoder,
            'labels': cfg.labels
        }

        threshold = 0.3

        results = []

        # STEP_LIST = [0.1, 0.15, 0.2]
        # WINDOW_SIZE_LIST = [0.5, 0.5, 0.5]
        STEP_LIST = [0.25]
        WINDOW_SIZE_LIST = [0.5]

        for STEP, WINDOW_SIZE in zip(STEP_LIST, WINDOW_SIZE_LIST, ):
            print(f'====== STEP is {STEP}s, WINDOW_SIZE is {WINDOW_SIZE}s ====== ')
            preds, proba_b, proba_s = offline_inference(enhance_wav_path, vad_model, model_definition, STEP, WINDOW_SIZE, threshold)
            results.append([STEP, WINDOW_SIZE, preds, proba_b, proba_s])

        # exit()
        num = len(results)
        speech_ranges = []

        for i in range(num):
            step = STEP_LIST[i]
            win_size = WINDOW_SIZE_LIST[i]
            len_pred = len(results[i][2])

            pred = results[i][2]

            last_label = 0
            start_pos = -0.1
            end_pos = -0.1

            count = 0
            for j, label in enumerate(pred):
                if label == 1 and last_label == 0:
                    start_pos = j
                    last_label = 1

                if label == 0 and last_label == 1:
                    end_pos = j
                    last_label = 0

                if start_pos >= 0 and end_pos > 0:
                    count += 1
                    print(f'({count}) speech : {start_pos * step:.2f} to {end_pos * step:.2f}')
                    speech_ranges.append((start_pos * step, end_pos * step))
                    start_pos = -0.1
                    end_pos = -0.1

        save_to_json(speech_ranges, 'vad_infer_result.json')

        last_start = -1
        last_end = 0

        idx = 0
        dialog_ranges = []

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

                if last_end - last_start > 30:
                    dialog_ranges.append((last_start, last_end))

                last_start = start_time
                last_end = end_time

                idx += 1
                pass
            else:
                # concat
                last_end = end_time

        if last_end - last_start > 30:
            dialog_ranges.append((last_start, last_end))

        print('=== dialog part ===')
        audio_list = []
        for i, dialog_range in enumerate(dialog_ranges):
            start_time = dialog_range[0]
            end_time = dialog_range[1]
            print(f'dialog({i}) : {convert(start_time)} - {convert(end_time)} (duration : {end_time - start_time})')

            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)
            audio, _ = soundfile.read(wav_path, start=start_frame, stop=end_frame, dtype='float32')
            audio_list.append(audio)

        #
        # Speech Recognition
        #
        model = whisper.load_model("medium")

        print('model loaded.')

        options = {
            'task': 'transcribe',
            'language': 'Korean'
        }

        text_list = []
        for audio in audio_list:
            result = model.transcribe(audio, **options)

            trans = []
            for segment in result['segments']:
                print(f'{segment["start"]} - {segment["end"]} : {segment["text"]}')
                trans.append(segment["text"])

            text_list.append(' '.join(trans))


        #
        # Threat Classifier
        #
        tokenizer = ElectraTokenizer.from_pretrained(
            '/root/sogang_asr/threat_model/baseline-kcelectra-newnew_train/tokenizer')
        model = ElectraForSequenceClassification.from_pretrained(
            '/root/sogang_asr/threat_model/baseline-kcelectra-newnew_train/epoch-26')  # 모델 경로 넣기

        predlist = []
        for text in text_list:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits

            predicted_class_id = logits.argmax().item()
            pred = model.config.id2label[predicted_class_id]
            predlist.append(pred)

        print(predlist)

    print('finished')


if __name__ == '__main__':
    filepaths = ['/root/sogang_asr/data/grand2022/cam1_all_noise_unique_v2.mp4']

    main(filepaths)