import imageio.v3 as iio
import numpy as np
import os


def combine_animation(record_dir, output_path, reverse):

    frames = []
    file_path_list = []
    for file_name in os.listdir(record_dir):
        if file_name.endswith('.gif'):
            file_path = os.path.join(record_dir, file_name)
            file_path_list.append(file_path)
    file_path_list.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    if reverse:
        file_path_list = file_path_list[::-1]

    for file_path in file_path_list:
        frame = iio.imread(file_path)
        if frame.shape[1] == 1536: # NOTE: hardcoded due to resolution differences between my screens
            frame = frame[:, ::2, ::2, :]
        frames.append(frame)

        print(file_path, len(frames))

    if len(frames) == 0:
        return

    frames = np.vstack(frames)

    # get duration each frame is displayed
    duration = iio.immeta(file_path)["duration"]

    output_dir = os.path.abspath(os.path.join(output_path, os.pardir))
    os.makedirs(output_dir, exist_ok=True)
    iio.imwrite(output_path, frames, duration=duration)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--record-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--reverse', default=False, action='store_true')
    args = parser.parse_args()

    combine_animation(args.record_dir, args.output_path, args.reverse)
    