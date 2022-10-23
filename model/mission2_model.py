import copy
import logging
import os
import tempfile
import time
import wave

import nemo.collections.asr as nemo_asr
import soundfile
import whisper
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import numpy as np
import torch
from espnet2.bin.enh_inference import SeparateSpeech
import onnxruntime as ort
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import ElectraTokenizer

from utils.asr_utils import ffmpeg_extract_wav, convert_to_16k, vad_post_process, convert


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text, label):
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


def create_examples(lines):
    examples = []
    for (i, line) in enumerate(lines):
        examples.append(InputExample(text=line, label=None))
    return examples


class SpeechEnhancement:
    def __init__(self, params=None, logger=None):
        self.device = f"cuda:{params['gpu']}"
        self.enh_model_sc = SeparateSpeech(
            train_config=params['config_path'],
            model_file=params['model_path'],
            # for segment-wise process on long speech
            normalize_segment_scale=False,
            show_progressbar=True,
            ref_channel=4,
            normalize_output_wav=True,
            device=self.device,
            segment_size=120,
            hop_size=96
        )

    def run(self, speech_mix, fs: int = 8000):
        return self.enh_model_sc(speech_mix, fs)


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
        audio_signal, audio_signal_len = audio_signal.to(self.vad_model.device), audio_signal_len.to(
            self.vad_model.device)
        logits = self.vad_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        return logits


class VAD:
    def __init__(self, params=None, logger=None):
        vad_model = nemo_asr.models.EncDecClassificationModel.restore_from(params['model_path'])
        self.device = f"cuda:{params['gpu']}"
        # Preserve a copy of the full config
        cfg = copy.deepcopy(vad_model._cfg)
        # print(OmegaConf.to_yaml(cfg))

        vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)
        # Set model to inference mode
        vad_model.eval()
        self.vad_model = vad_model.to(self.device)

        self.vad_model_definition = {
            'sample_rate': cfg.train_ds.sample_rate,
            'AudioToMFCCPreprocessor': cfg.preprocessor,
            'JasperEncoder': cfg.encoder,
            'labels': cfg.labels
        }

        self.window_size = params['window_size']
        self.frame_len = params['frame_len']
        self.threshold = params['threshold']

    def offline_inference(self, wave_file, step=0.025, window_size=0.5, threshold=0.5):
        print(f'====== FRAME_LEN is {step}s, WINDOW_SIZE is {window_size}s ====== ')
        frame_len = step  # infer every STEP seconds
        channels = 1  # number of audio channels (expect mono signal)
        rate = 16000  # sample rate, Hz

        chunk_size = int(frame_len * rate)

        vad = FrameVAD(vad_model=self.vad_model,
                       model_definition=self.vad_model_definition,
                       threshold=threshold,
                       frame_len=frame_len, frame_overlap=(window_size - frame_len) / 2,
                       offset=0)

        wf = wave.open(wave_file, 'rb')

        empty_counter = 0

        preds = []
        proba_b = []
        proba_s = []

        data = wf.readframes(chunk_size)

        while len(data) > 0:

            data = wf.readframes(chunk_size)
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

    def inference(self, wave_file):
        preds, proba_b, proba_s = self.offline_inference(wave_file,
                                                         step=self.frame_len,
                                                         window_size=self.window_size,
                                                         threshold=self.threshold)

        speech_ranges = []

        last_label = 0
        start_pos = -0.1
        end_pos = -0.1

        count = 0
        for j, label in enumerate(preds):
            if label == 1 and last_label == 0:
                start_pos = j
                last_label = 1

            if label == 0 and last_label == 1:
                end_pos = j
                last_label = 0

            if start_pos >= 0 and end_pos > 0:
                count += 1
                print(f'({count}) speech : {start_pos * self.frame_len:.2f} to {end_pos * self.frame_len:.2f}')
                speech_ranges.append((start_pos * self.frame_len, end_pos * self.frame_len))
                start_pos = -0.1
                end_pos = -0.1

        return speech_ranges


class SpeechRecognition():
    def __init__(self, params=None, logger=None):
        self.asr_model = whisper.load_model(params['model_path'])

        self.whisper_options = {
            'task': 'transcribe',
            'language': 'Korean'
        }

    def transcribe(self, audio):
        return self.asr_model.transcribe(audio, **self.whisper_options)


class ThreatClassification():
    def __init__(self, params=None, logger=None):
        # ONNX model inference
        self.device = f"cuda:{params['gpu']}"
        self.id2label = {0: '020121', 1: '000001', 2: '02051', 3: '020811', 4: '020819'}
        self.tokenizer = ElectraTokenizer.from_pretrained(params['tokenizer_path'])

        opt = ort.SessionOptions()
        EP_list = [
            ('CUDAExecutionProvider', {
                'device_id': params['gpu']
            })
        ]
        self.session = ort.InferenceSession(params['model_path'], opt, providers=EP_list)

    def inference(self, text_list):
        test_dataset = self.load_data(text_list)
        results = self.evaluate(test_dataset)
        return results

    def evaluate(self, eval_dataset):
        eval_batch_size = 64

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        logging.debug("***** Running evaluation *****")
        logging.debug("  Num examples = {}".format(len(eval_dataset)))
        logging.debug("  Eval Batch size = {}".format(eval_batch_size))

        preds = None

        for batch in eval_dataloader:

            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": np.atleast_2d(batch[0].detach().cpu().numpy()),
                    "attention_mask": np.atleast_2d(batch[1].detach().cpu().numpy()),
                    "token_type_ids": np.atleast_2d(batch[2].detach().cpu().numpy()),
                }

                outputs = self.session.run(None, inputs)
                logits = outputs[0]

            if preds is None:
                preds = logits
            else:
                preds = np.append(preds, logits, axis=0)

        preds = np.argmax(preds, axis=1)

        pred_list = []
        for pred in preds:
            pred_list.append(self.id2label[pred])

        return pred_list

    def load_data(self, data):
        examples = create_examples(data)
        features = self.seq_cls_convert_examples_to_features(examples)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        # all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

        return dataset

    def seq_cls_convert_examples_to_features(self, examples):
        max_length = 512

        batch_encoding = self.tokenizer.batch_encode_plus(
            [str(example.text) for example in examples],
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            if "token_type_ids" not in inputs:
                inputs["token_type_ids"] = [0] * len(inputs["input_ids"])
            feature = InputFeatures(**inputs, label=None)
            features.append(feature)

        for i, example in enumerate(examples[:5]):
            logging.debug("*** Example ***")
            logging.debug("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
            logging.debug("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
            logging.debug("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
            logging.debug("label: {}".format(features[i].label))

        return features


class Mission2Manager():
    def __init__(self, params):
        self.global_start_time = time.time()
        self.global_running_time = None
        self.total_audio_duration = 0.0
        self.time_dict = {}
        self.model_params = params["model"]

        self.enh_model = None
        self.vad_model = None
        self.asr_model = None
        self.nlp_model = None

    def load_model(self):
        if not self.enh_model:
            self.load_speech_enhancement_model()
        if not self.vad_model:
            self.load_voice_activity_model()
        if not self.asr_model:
            self.load_speech_recognition_model()
        if not self.nlp_model:
            self.load_threat_classification_model()

    def load_speech_enhancement_model(self):
        #
        # Load Model: Speech Enhancement
        #
        logging.info('Loading - Speech Enhancement Model')

        model_start_time = time.time()

        self.enh_model = SpeechEnhancement(self.model_params['speech_enhancement'])

        model_loading_time = time.time() - model_start_time

        self.time_dict['enh'] = {
            'model_loading': model_loading_time,
            'model_running': 0.0
        }

    def load_voice_activity_model(self):
        #
        # Load Model: VAD
        #
        logging.info('Loading - Voice Activity Detection Model')

        model_start_time = time.time()

        self.vad_model = VAD(self.model_params['vad'])

        model_loading_time = time.time() - model_start_time

        self.time_dict['vad'] = {
            'model_loading': model_loading_time,
            'model_running': 0.0,
        }

    def load_speech_recognition_model(self):
        #
        # Load Model: Speech Recognition
        #
        logging.info('Loading - Speech Recognition Model')

        model_start_time = time.time()

        self.asr_model = SpeechRecognition(self.model_params['speech_recognition'])

        model_loading_time = time.time() - model_start_time

        self.time_dict['asr'] = {
            'model_loading': model_loading_time,
            'model_running': 0.0
        }

    def load_threat_classification_model(self):
        #
        # Load Model: Threat Classifier
        #
        logging.info('Loading - Threat Classification Model')

        model_start_time = time.time()

        self.nlp_model = ThreatClassification(self.model_params['threat_classification'])

        model_loading_time = time.time() - model_start_time

        self.time_dict['nlp'] = {
            'model_loading': model_loading_time,
            'model_running': 0.0
        }

    def run_mission2(self, file_path, file_type='video'):
        wave_paths = []

        self.time_dict['pre'] = {
            'model_running': 0.0
        }

        t = tempfile.TemporaryDirectory()
        out_path = t.name

        if file_type == 'video':
            #
            # mp4 to wav
            #
            model_start_time = time.time()

            print(f'processing : {file_path}')

            filename, file_extension = os.path.splitext(os.path.basename(file_path))
            wav_path = os.path.join(out_path, f'{filename}.wav')
            ffmpeg_extract_wav(file_path, wav_path)
            wave_paths.append(wav_path)

            model_running_time = time.time() - model_start_time

            self.time_dict['pre']['model_running'] += model_running_time

        else:
            wave_paths.append(convert_to_16k(file_path, out_path))

        for wav_path in wave_paths:
            #
            # Speech Enhancement
            #
            model_start_time = time.time()

            mixwav_mc, sr = soundfile.read(wav_path)
            # mixwav.shape: num_samples, num_channels
            # mixwav_sc = mixwav_mc[:,4]
            mixwav_sc = mixwav_mc[:]

            self.total_audio_duration += len(mixwav_mc) / float(sr)

            filename, file_extension = os.path.splitext(os.path.basename(wav_path))
            enhance_wav_path = os.path.join(out_path, f'{filename}_enhance.wav')

            wave = self.enh_model.run(mixwav_sc[None, ...], sr)
            soundfile.write(enhance_wav_path, wave[0].squeeze(), sr)

            model_running_time = time.time() - model_start_time

            self.time_dict['enh']['model_running'] += model_running_time

            #
            # VAD
            #
            model_start_time = time.time()

            speech_ranges = self.vad_model.inference(enhance_wav_path)

            # save_to_json(speech_ranges, 'vad_infer_result.json')

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

            self.time_dict['vad']['model_running'] += model_running_time

            #
            # Speech Recognition
            #
            model_start_time = time.time()

            text_list = []
            for audio in audio_list:
                result = self.asr_model.transcribe(audio[0])

                trans = []
                for segment in result['segments']:
                    print(f'{segment["start"]} - {segment["end"]} : {segment["text"]}')
                    trans.append(segment["text"])

                text_list.append(' '.join(trans))

            model_running_time = time.time() - model_start_time

            self.time_dict['asr']['model_running'] += model_running_time

            #
            # Threat Classifier
            #
            model_start_time = time.time()

            pred_list = self.nlp_model.inference(text_list)
            print(pred_list)

            model_running_time = time.time() - model_start_time

            self.time_dict['nlp']['model_running'] += model_running_time

        print('\n=== Final results ===')
        out_list = []
        for audio, pred_label in zip(audio_list, pred_list):
            start_time = audio[1]
            end_time = audio[2]
            print(f'{start_time} - {end_time} : {pred_label}')
            out_list.append((start_time, end_time, pred_label))

        return out_list

    def end_mission(self):
        self.global_running_time = time.time() - self.global_start_time

    def print_statistics(self):
        if not self.global_running_time:
            self.end_mission()

        print('\n=== Statistics ===')
        for module, elem in self.time_dict.items():
            print(f'{module} : {elem}')

        print(f'\ntotal audio duration : {self.total_audio_duration:.2f}')
        print(f'total running time : {self.global_running_time:.2f}')
        print(f'RTF : {self.global_running_time / self.total_audio_duration:.4f}')