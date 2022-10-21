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
    audio_result_file = 'results/mission2_result.json'

    video_fps = 15

    video_results = load_json(video_result_file)
    audio_results = load_json(audio_result_file)

    mission_info = video_results['mission_info']
    frame_results = video_results['frame_results']
    final_result = video_results['final_result']

    assert mission_info['mission'] == 'mission1'

    initial_info = mission_info['initial_info']
    info_date = initial_info['date'] #비디오의 날짜
    info_start_time = initial_info['start_time'] #비디오의 시작 시간
    info_cam = initial_info['cam']

    info_mission = mission_info['mission']

    start_datetime = datetime.strptime(f'{info_date} {info_start_time}', "%Y/%m/%d %H:%M:%S") #비디오의 시각 날짜, 시간

    print(start_datetime)

    for audio_result in audio_results: #대화가 있는 구간, 그 구간의 라벨
        answer_sheet = {}

        start_time = audio_result[0] #대화의 시작시각
        end_time = audio_result[1] #대화의 끝나는 시각
        threat_label = audio_result[2] #대화의 라벨

        audio_start = start_datetime + timedelta(seconds=start_time) #하나의 대화의 시작 시각 (비디오의 날짜 시각과 통합)
        audio_end = start_datetime + timedelta(seconds=end_time)  # 하나의 대화의 끝나는 시각 (비디오 날짜 시각과 통합)

        print(audio_result)
        print(audio_start)
        print(audio_end)

        time_start = audio_start.strftime('%H:%M:%S') #대화가 시작하는 시각
        time_end = audio_end.strftime('%H:%M:%S') #대화가 끝나는 시각

        # time to frame
        start_frame = start_time * video_fps
        end_frame = end_time * video_fps

        print(f'start_frame : {start_frame}') #대화 시작 시각의 비디오 frame number
        print(f'end_frame : {end_frame}') #대화 끝나는 시각의 비디오 frame number

        age_gender_dict = {}
        people_dict = {}

        people_count = 0
        group_array = {}

        frame_count = len(frame_results) #총 frame results의 숫자

        start_frame = start_frame + start_frame//3
        end_frame = end_frame - end_frame % 3
        
        for idx, frame_result in enumerate(frame_results):
            if frame_result["frame_number"]==start_frame:
                start_idx=idx
            if frame_result["frame_number"]==end_frame:
                end_idx=idx

        for count, frame_result in enumerate(frame_results):
            one_before_group_array={}
            assert len(frame_result.keys()) == 2 #frame number, result

            frame_number = frame_result['frame_number']

            if frame_number < start_frame or frame_number > end_frame: # 대화 구간에 해당하는 프레임이 아닌 경우 패스
                continue

            print(f'video time : {frame_number / video_fps}') #비디오 시각으로 표현

            results = frame_result['result'] #객체의 숫자만큼 result가 나옴

            result_len = len(results) #객체의 숫자
            #print(f'result len : {result_len}')

            if result_len < 2: #객체의 수가 2명이상이 아닌 경우 패스 (그룹이 아니므로 검출할 필요 없음)
                continue

            position_data = []
            position_x = []
            position_y = []

            for i, result in enumerate(results): # 한 명의 사람 result 결과에 대해
                print(result)
                assert len(result.keys()) == 3 #label, position, pose

                label = result['label'] #객체의 종류가 무엇인지 확인해야함. person인 경우 age, gender가 검출됨
                if not label["class"] == "person": #사람이 아니면 뽑을 필요가 없음
                    continue

                position = result['position'] #활용해야 할 부분
                pose = result['pose']

                position_x.append(position['x']) #각 인덱스는 사람의 인덱스, 그 사람의 x좌표
                position_y.append(position['y'])
                position_data.append((position['x'], position['y']))


                age_gender = label['age_gender']
                age_gender_class = age_gender['class']
                age_gender_score = age_gender['score']

                count_dict(data_dict=age_gender_dict, key=age_gender_class)
            count_dict(data_dict=people_dict, key=result_len)

            print(f'people num: {len(position_x)}') #하나의 프레임에 있는 사람의 수
            print(position_data) #사람들의 위치 데이터

            max_x = max(position_x)
            min_x = min(position_x)
            max_idx = position_x.index(max_x)
            min_idx = position_x.index(min_x)

            group_num = 0
            group_list = [-1] * len(position_x)

            diff = 200

            for i, x in enumerate(position_x):
                if x - min_x < diff:
                    group_list[i] = group_num

                    try:
                        one_before_group_array[group_num].append(x)
                    except KeyError:
                        one_before_group_array[group_num] = [x]

            group_num += 1

            for i, x in enumerate(position_x):
                if max_x - x < diff:
                    group_list[i] = group_num

                    try:
                        one_before_group_array[group_num].append(x)
                    except KeyError:
                        one_before_group_array[group_num] = [x]

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
                            one_before_group_array[group_num].append(x)
                        except KeyError:
                            one_before_group_array[group_num] = [x]

            assert -1 not in group_list
            print(group_list)
            print(f'one_before_group_array: {one_before_group_array}')
            print(f'group_array: {group_array}')

            if count==start_idx:
                for key, value in one_before_group_array.items():
                    group_array[key]=value
                continue

            # 이전 프레임의 그룹 넘버와 일치시키기
            local_group_array={}
            group_len=max(set(group_list))

            for i in range(group_len+1):
                local_group_list=list(filter(lambda x: group_list[x] == i, range(len(group_list))))
                for idx in local_group_list:
                    try:
                        local_group_array[i].append(position_x[idx])
                    except KeyError:
                        local_group_array[i] = [position_x[idx]]
            print(f'local_group_array: {local_group_array}')

            group_mean_array={}
            one_before_group_mean_array={}

            for key, value in local_group_array.items():
                group_mean_array[key]=np.array(value).mean()

            for key, value in one_before_group_array.items():
                one_before_group_mean_array[key]=np.array(value).mean()

            print(f'group_mean_array: {group_mean_array}')
            print(f'one_before_group_mean_array: {one_before_group_mean_array}')

            for key, value in group_mean_array.items():
                min=100000
                min_idx=-1
                for k, v in one_before_group_mean_array.items():
                    if abs(value-v) < min:
                        min = abs(value-v)
                        min_idx=k
                try:
                    group_array[min_idx].append(x)
                except KeyError:
                    group_array[min_idx] = [x]

            print(f'group_array: {group_array}')
            gr


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
