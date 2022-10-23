import logging

from model.post.submission import MissionSubmission
from utils.utils import load_json

TEAM_ID = 'convai'
SECRET = '3dlZhXRPPyt22tR9'


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    video_result_file = 'results/cam1_short_result.json'
    audio_result_file = 'results/cam1_short_result_m2.json'

    video_results = load_json(video_result_file)
    audio_results = load_json(audio_result_file)

    submission = MissionSubmission(team_id=TEAM_ID, secret=SECRET)

    results = submission.postprocess_mission2(video_results, audio_results)

    for result in results:
        print(result)
