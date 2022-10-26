import yaml
import logging

from model.mission2_model import Mission2Manager
from utils.utils import save_to_json


def load_settings(settings_path):
    with open(settings_path) as settings_file:
        settings = yaml.safe_load(settings_file)
    return settings


def main(filepaths, params, file_type='video'):
    manager = Mission2Manager(params)
    manager.load_model()

    for filepath in filepaths:
        out_list = manager.run_mission2(file_path=filepath, file_type=file_type)
        save_to_json(out_list, 'mission2_result.json')

    manager.print_statistics()
    print('finished')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    params_path = 'config/params.yaml'
    params = load_settings(params_path)
    filepaths = [
        '/root/sogang_asr/data/grand2022/cam1_short.mp4',
        '/root/sogang_asr/data/grand2022/cam1_all_noise_unique_v2.mp4',
        # '/root/sogang_asr/data/grand2022/cam1_all_noise_30min_v2.mp4',
    ]
    main(filepaths, params, file_type='video')

    # filepaths = [
    #     '/root/sogang_asr/data/grand2022/speech_noise_mixdown_004.wav'
    # ]
    #
    # main(filepaths, params, file_type='audio')
