import copy
import wave

import nemo.collections.asr as nemo_asr
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import numpy as np
import torch
from torch.utils.data import DataLoader
from espnet2.bin.enh_inference import SeparateSpeech


class SpeechEnhancement:
    def __init__(self, params=None, logger=None):

        self.enh_model_sc = SeparateSpeech(
            train_config=params['config_path'],
            model_file=params['model_path'],
            # for segment-wise process on long speech
            normalize_segment_scale=False,
            show_progressbar=True,
            ref_channel=4,
            normalize_output_wav=True,
            device="cuda:0",
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
        audio_signal, audio_signal_len = audio_signal.to(self.vad_model.device), audio_signal_len.to(self.vad_model.device)
        logits = self.vad_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        return logits


class VAD:
    def __init__(self, params=None, logger=None):
        vad_model = nemo_asr.models.EncDecClassificationModel.restore_from(params['model_path'])
        # Preserve a copy of the full config
        cfg = copy.deepcopy(vad_model._cfg)
        # print(OmegaConf.to_yaml(cfg))

        vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)
        # Set model to inference mode
        vad_model.eval()
        self.vad_model = vad_model.to(vad_model.device)

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
        preds, proba_b, proba_s =  self.offline_inference(wave_file,
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