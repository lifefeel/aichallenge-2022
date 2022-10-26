import logging
import os
from pprint import pprint

from model.post.submission import MissionSubmission
from utils.utils import load_json

TEAM_ID = '1st_isds'
SECRET = 'FaAJkyJuzlPsFk3f'


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    api_url = os.environ.get('REST_ANSWER_URL', 'http://163.239.14.105:5000')

    video_result_file = 'results/cam1_short_result.json'
    audio_result_file = 'results/cam1_short_result_m2.json'

    video_results = load_json(video_result_file)
    audio_results = load_json(audio_result_file)

    submission = MissionSubmission(team_id=TEAM_ID, secret=SECRET, api_url=api_url)

    results = submission.postprocess_mission1(video_results)

    for result in results:
        pprint(result)
        # submission.submit(result)

    # exit()

    results = submission.postprocess_mission2_only(video_results['mission_info'], audio_results)

    for result in results:
        pprint(result)
        submission.submit(result)

    submission.end_of_mission()