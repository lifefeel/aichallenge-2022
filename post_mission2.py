from datetime import datetime, timedelta

from utils.utils import load_json

TEAM_ID = 'convai'
SECRET = '3dlZhXRPPyt22tR9'


def generate_answer_sheet(cam_no, mission, answer):
    return {
        'team_id': TEAM_ID,
        'secret': SECRET,
        'answer_sheet': {
            'cam_no': str(cam_no),  # TODO 숫자만 들어가도록 처리 필요 (예: "03")
            'mission': str(mission),  # TODO 숫자만 들어가도록 처리 필요 (예: "2")
            'answer': answer
        }
    }


if __name__ == '__main__':
    video_result_file = 'results/cam1_short_result.json'
    audio_result_file = 'mission2_result.json'

    video_fps = 15

    video_results = load_json(video_result_file)
    audio_results = load_json(audio_result_file)

    mission_info = video_results['mission_info']
    frame_results = video_results['frame_results']
    final_result = video_results['final_result']

    assert mission_info['mission'] == 'mission1'

    initial_info = mission_info['initial_info']
    info_date = initial_info['date']
    info_start_time = initial_info['start_time']
    info_cam = initial_info['cam']

    info_mission = mission_info['mission']

    start_datetime = datetime.strptime(f'{info_date} {info_start_time}', "%Y/%m/%d %H:%M:%S")

    print(start_datetime)

    for audio_result in audio_results:
        answer_sheet = {}

        start_time = audio_result[0]
        end_time = audio_result[1]
        label = audio_result[2]

        audio_start = start_datetime + timedelta(seconds=start_time)
        audio_end = start_datetime + timedelta(seconds=end_time)

        print(audio_result)
        print(audio_start)
        print(audio_end)

        time_start = audio_start.strftime('%H:%M:%S')
        time_end = audio_end.strftime('%H:%M:%S')

        # time to frame
        start_frame = start_time * video_fps
        end_frame = end_time * video_fps

        print(f'start_frame : {start_frame}')
        print(f'end_frame : {end_frame}')

        for frame_result in frame_results:
            assert len(frame_result.keys()) == 2

            frame_number = frame_result['frame_number']

            if frame_number < start_frame or frame_number > end_frame:
                continue

            print(f'video time : {frame_number / video_fps}')

            results = frame_result['result']

            print(f'result len : {len(results)}')
            for result in results:
                print(result)

            # print(frame_result)

        answer = {
            'event': label,
            'time_start': time_start,
            'time_end': time_end
        }

        print(generate_answer_sheet(cam_no=info_cam, mission=info_mission, answer=answer))










