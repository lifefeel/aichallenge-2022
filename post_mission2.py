from datetime import datetime, timedelta
import numpy as np

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


def count_dict(data_dict, key):
    try:
        data_dict[key] += 1
    except KeyError:
        data_dict[key] = 1


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
        threat_label = audio_result[2]

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

        age_gender_dict = {}
        people_dict = {}

        people_count = 0
        group_array = {}

        frame_count = len(frame_results)

        for frame_result in frame_results:
            assert len(frame_result.keys()) == 2

            frame_number = frame_result['frame_number']

            if frame_number < start_frame or frame_number > end_frame:
                continue

            print(f'video time : {frame_number / video_fps}')

            results = frame_result['result']

            result_len = len(results)
            print(f'result len : {result_len}')

            position_data = []
            position_x = []
            position_y = []

            for i, result in enumerate(results):
                print(result)

                assert len(result.keys()) == 3

                label = result['label']
                position = result['position']
                pose = result['pose']

                position_x.append(position['x'])
                position_y.append(position['y'])
                position_data.append((position['x'], position['y']))

                age_gender = label['age_gender']
                age_gender_class = age_gender['class']
                age_gender_score = age_gender['score']

                count_dict(data_dict=age_gender_dict, key=age_gender_class)
            count_dict(data_dict=people_dict, key=result_len)

            if len(position_data) < 2:
                continue

            print(position_data)

            max_x = max(position_x)
            min_x = min(position_x)
            max_idx = position_x.index(max_x)
            min_idx = position_x.index(min_x)

            group_num = 0
            group_list = [-1] * len(position_x)

            diff = 500

            for i, x in enumerate(position_x):
                if x - min_x < diff:
                    group_list[i] = group_num

                    try:
                        group_array[group_num].append(x)
                    except KeyError:
                        group_array[group_num] = [x]

            group_num += 1

            for i, x in enumerate(position_x):
                if max_x - x < diff:
                    group_list[i] = group_num

                    try:
                        group_array[group_num].append(x)
                    except KeyError:
                        group_array[group_num] = [x]

            if -1 in group_list:
                min_x2 = 10000
                for i, x in enumerate(position_x):
                    if group_list[i] != -1:
                        continue

                    if x < min_x2:
                        min_x2 = x

                group_num += 1

                for i, x in enumerate(position_x):
                    if group_list[i] != -1:
                        continue

                    if x - min_x2 < diff:
                        group_list[i] = group_num

                        try:
                            group_array[group_num].append(x)
                        except KeyError:
                            group_array[group_num] = [x]

            assert -1 not in group_list
            print(group_list)

        answer = {
            'event': threat_label,
            'time_start': time_start,
            'time_end': time_end
        }

        min_std_idx = -1
        min_std = 10000

        for key, val in group_array.items():
            if len(val) < frame_count * 0.5:
                continue

            data_x = np.array(val)
            print(key, val)
            print(f'mean : {data_x.mean()}')
            print(f'var : {data_x.var()}')

            std = data_x.std()
            print(f'std : {std}')

            if std < min_std:
                min_std = std
                min_std_idx = key


        print(f'min_std_group : {min_std_idx}')




        print('=== Result ===')
        print(f'num people : {people_count}')
        print(age_gender_dict)
        print(people_dict)
        print(generate_answer_sheet(cam_no=info_cam, mission=info_mission, answer=answer))

    print(f'last frame time : {frame_results[-1]["frame_number"] / video_fps}')
