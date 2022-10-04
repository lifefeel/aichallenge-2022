# This is a sample Python script.
import ffmpeg
import os

def ffmpeg_extract_wav(input_path, output_path):
    input_stream = ffmpeg.input(input_path)

    output_wav = ffmpeg.output(input_stream.audio, output_path + ".wav", acodec='pcm_s16le', ac=1, ar='48k')
    output_wav.overwrite_output().run()


if __name__ == '__main__':

    # dir_path = '/root/sogang_asr/data/movie'
    dir_path = '/root/sogang_asr/data/grand2022/sample_cam1'

    for curdir, dirs, files in os.walk(dir_path):
        for file in os.listdir(curdir):
            filename, file_extension = os.path.splitext(file)
            if file_extension != '.mp4':
                continue

            in_file = os.path.join(curdir, file)
            out_file = os.path.join(curdir, filename)
            ffmpeg_extract_wav(in_file, out_file)