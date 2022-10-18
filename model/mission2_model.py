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

