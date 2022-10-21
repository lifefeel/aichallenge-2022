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


def count_dictionary(data_dict, key):
    try:
        data_dict[key] += 1
    except KeyError:
        data_dict[key] = 1


def argmax_dict(data_dict):
    max_val = -1
    max_idx = -1
    for key, val in data_dict.items():
        if val > max_val:
            max_idx = key
            max_val = val

    return max_idx


def count_agegender(data_list):
    gender_dict = {}
    for elem in data_list:
        elem = sorted(elem)
        gender_str = '_'.join(elem)

        count_dictionary(gender_dict, gender_str)

    print(gender_dict)
    print(argmax_dict(gender_dict))

    max_gender_str = argmax_dict(gender_dict)
    return max_gender_str.split('_')


if __name__ == '__main__':
    video_result_file = 'results/cam1_short_result.json'
    audio_result_file = 'results/mission2_result.json'

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

        people_count = 0
        group_x_array = {}
        group_count = {
            0: {},
            1: {},
            2: {},
            3: {},
        }

        gender_count = {
            0: {},
            1: {},
            2: {},
            3: {},
        }

        total_frame_count = len(frame_results)
        related_frame_count = 0

        for frame_result in frame_results:
            assert len(frame_result.keys()) == 2

            frame_number = frame_result['frame_number']

            if frame_number < start_frame or frame_number > end_frame:
                continue

            related_frame_count += 1

            print('========================================')
            print(f'video frame: {frame_number} (time: {frame_number / video_fps})')

            results = frame_result['result']

            result_len = len(results)
            print(f'object result len : {result_len}')

            position_data = []
            position_x = []
            position_y = []
            local_group_dict = {}
            local_age_gender_dict = {}

            for i, result in enumerate(results):
                assert len(result.keys()) == 3

                label = result['label']
                position = result['position']
                pose = result['pose']

                age_gender = label['age_gender']
                age_gender_class = age_gender['class']
                age_gender_score = age_gender['score']

                if age_gender_class == 'Unknown':  # 사람이 아닌 경우 넘어감
                    continue

                position_x.append(position['x'])
                position_y.append(position['y'])
                position_data.append((position['x'], position['y']))

                print(result)

            if len(position_data) < 2:
                continue

            print(f'position_data: {position_data}')

            max_x = max(position_x)
            min_x = min(position_x)
            max_idx = position_x.index(max_x)
            min_idx = position_x.index(min_x)

            group_num = 0
            group_list = [-1] * len(position_x)

            diff = 500

            # 가장 작은 값과 diff 만큼 차이나는 것을 같은 그룹으로 묶음
            if min_x <= 500:
                group_num = 0
                for i, x in enumerate(position_x):
                    if x - min_x < diff:
                        group_list[i] = group_num

                        try:
                            group_x_array[group_num].append(x)
                        except KeyError:
                            group_x_array[group_num] = [x]

                        try:
                            local_group_dict[group_num].append(x)
                        except KeyError:
                            local_group_dict[group_num] = [x]

                        try:
                            local_age_gender_dict[group_num].append(age_gender_class)
                        except KeyError:
                            local_age_gender_dict[group_num] = [age_gender_class]

            if max_x >= 1420:
                group_num = 1

                # 가작 큰 값과 diff 만큼 차이나는 것을 같은 그룹으로 묶음
                for i, x in enumerate(position_x):
                    if max_x - x < diff:
                        group_list[i] = group_num

                        try:
                            group_x_array[group_num].append(x)
                        except KeyError:
                            group_x_array[group_num] = [x]

                        try:
                            local_group_dict[group_num].append(x)
                        except KeyError:
                            local_group_dict[group_num] = [x]

                        try:
                            local_age_gender_dict[group_num].append(age_gender_class)
                        except KeyError:
                            local_age_gender_dict[group_num] = [age_gender_class]

            # 처리가 안 된 객체가 있는 경우(중앙 좌측)
            if -1 in group_list:
                group_num = 2
                min_x2 = 10000
                for i, x in enumerate(position_x):
                    if group_list[i] != -1:
                        continue

                    if x < min_x2:
                        min_x2 = x

                for i, x in enumerate(position_x):
                    if group_list[i] != -1:
                        continue

                    if x - min_x2 < diff:
                        group_list[i] = group_num

                        try:
                            group_x_array[group_num].append(x)
                        except KeyError:
                            group_x_array[group_num] = [x]

                        try:
                            local_group_dict[group_num].append(x)
                        except KeyError:
                            local_group_dict[group_num] = [x]

                        try:
                            local_age_gender_dict[group_num].append(age_gender_class)
                        except KeyError:
                            local_age_gender_dict[group_num] = [age_gender_class]

            # 처리가 안 된 객체가 있는 경우(중앙 우측)
            if -1 in group_list:
                group_num = 3
                max_x2 = 0
                for i, x in enumerate(position_x):
                    if group_list[i] != -1:
                        continue

                    if x > max_x2:
                        max_x2 = x

                for i, x in enumerate(position_x):
                    if group_list[i] != -1:
                        continue

                    if x - max_x2 < diff:
                        group_list[i] = group_num

                        try:
                            group_x_array[group_num].append(x)
                        except KeyError:
                            group_x_array[group_num] = [x]

                        try:
                            local_group_dict[group_num].append(x)
                        except KeyError:
                            local_group_dict[group_num] = [x]

                        try:
                            local_age_gender_dict[group_num].append(age_gender_class)
                        except KeyError:
                            local_age_gender_dict[group_num] = [age_gender_class]

            assert -1 not in group_list
            print(f'group_list: {group_list}')
            print(f'local_group_dict: {local_group_dict}')
            print(f'local_age_gender_dict: {local_age_gender_dict}')

            for idx, x_list in local_group_dict.items():
                people_len = len(x_list)
                try:
                    group_count[idx][people_len] += 1
                except KeyError:
                    group_count[idx][people_len] = 1


            for idx, x_list in local_age_gender_dict.items():
                people_len = len(x_list)
                try:
                    gender_count[idx][people_len].append(x_list)
                except KeyError:
                    gender_count[idx][people_len] = [x_list]

        print(f'group_count : {group_count}')

        answer = {
            'event': threat_label,
            'time_start': time_start,
            'time_end': time_end
        }

        min_std_idx = -1
        min_std = 10000

        for key, val in group_x_array.items():
            if len(val) < total_frame_count * 0.5:  # 해당 구간에 일정 비율 이하로 등장하는 사람그룹은 제외
                continue

            data_x = np.array(val)

            print(f'group {key}: ')
            print(f'mean : {data_x.mean()}')
            print(f'var : {data_x.var()}')

            std = data_x.std()
            print(f'std : {std}')

            if std < min_std:
                min_std = std
                min_std_idx = key

        print(f'min_std_group : {min_std_idx}')
        num_people = argmax_dict(group_count[min_std_idx])
        print(gender_count[min_std_idx][num_people])

        age_gender_list = count_agegender(gender_count[min_std_idx][num_people])

        print('=== Result ===')
        print(f'total frame count: {total_frame_count}')
        print(f'related frame count: {related_frame_count}')
        print(f'num people: {num_people}')
        print(f'age gender: {age_gender_list}')
        print(generate_answer_sheet(cam_no=info_cam, mission=info_mission, answer=answer))

    print(f'last frame time : {frame_results[-1]["frame_number"] / video_fps}')
