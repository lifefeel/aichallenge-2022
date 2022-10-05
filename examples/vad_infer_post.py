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

print('=== chunking ===')
for i, speech_range in enumerate(speech_ranges):
    start_time = speech_range[0]
    end_time = speech_range[1]

    # print(f'({i}) speech : {start_time} to {end_time}')

    if last_start < 0:
        last_start = start_time
        last_end = end_time

    if start_time - last_end > 5:
        # new chunk
        print(f'  chunk({idx}) : {convert(last_start)} - {convert(last_end)} (duration : {last_end - last_start})')

        if last_end - last_start > 30:
            dialog_ranges.append((last_start, last_end))

        last_start = start_time
        last_end = end_time

        idx += 1
        pass
    else:
        # concat
        last_end = end_time

if last_end - last_start > 30:
    dialog_ranges.append((last_start, last_end))

print('=== dialog part ===')
for i, dialog_range in enumerate(dialog_ranges):
    start_time = dialog_range[0]
    end_time = dialog_range[1]
    print(f'dialog({i}) : {convert(start_time)} - {convert(end_time)}')