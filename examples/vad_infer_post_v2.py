from utils.utils import load_json


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    milli_sec = sec % 1 * 1000
    return '%02d:%02d:%02d.%03d' % (hour, min, sec, milli_sec)


speech_ranges = load_json('vad_infer_result.json')

last_start = -1
last_end = 0

idx = 0
dialog_ranges = []
long_ranges = []
dialog_min_length = 30
long_speech_threshold = 0.5

print('=== long range ===')
for i, speech_range in enumerate(speech_ranges):
    start_time = speech_range[0]
    end_time = speech_range[1]

    # print(f'({i}) speech : {start_time} to {end_time}')
    if end_time - start_time >= long_speech_threshold:
        print(f'long speech : {convert(start_time)} - {convert(end_time)} ({start_time} to {end_time})')
        long_ranges.append((i, start_time, end_time))


new_ranges = []
for long_range in long_ranges:
    idx, start_time, end_time = long_range
    i = 0
    new_start = start_time
    while True:
        i += 1

        if idx - i < 0:
            print('pre index finished.')
            break

        pre_start, pre_end = speech_ranges[idx - i]

        if new_start - pre_end >= 2.5:
            break

        new_start = pre_start

    i = 0
    new_end = end_time
    while True:
        i += 1

        try:
            next_start, next_end = speech_ranges[idx + i]
        except IndexError:
            print('next index finished.')
            break

        if next_start - new_end >= 2.5:
            break

        new_end = next_end

    if start_time != new_start or end_time != new_end:
        new_range = (new_start, new_end)
        if new_range not in new_ranges:
            new_ranges.append(new_range)

for new_range in new_ranges:
    start_time, end_time = new_range
    print(f'new range : {convert(start_time)} - {convert(end_time)} ({start_time} to {end_time})')

print('=== chunking ===')
for i, speech_range in enumerate(new_ranges):
    start_time, end_time = speech_range

    if last_start < 0:
        last_start = start_time
        last_end = end_time

    # if end_time - start_time < 5:
    #     continue

    if start_time - last_end > 5:
        # new chunk
        print(f'  chunk({idx}) : {convert(last_start)} - {convert(last_end)} (duration : {last_end - last_start})')

        if last_end - last_start > dialog_min_length:
            dialog_ranges.append((last_start, last_end))

        last_start = start_time
        last_end = end_time

        idx += 1
        pass
    else:
        # concat
        last_end = end_time

print(f'  chunk({idx}) : {convert(last_start)} - {convert(last_end)} (duration : {last_end - last_start})')

if last_end - last_start > dialog_min_length:
    dialog_ranges.append((last_start, last_end))

print('=== dialog part ===')
for i, dialog_range in enumerate(dialog_ranges):
    start_time = dialog_range[0]
    end_time = dialog_range[1]
    print(f'dialog({i}) : {convert(start_time)} - {convert(end_time)} (duration : {end_time - start_time})')
