import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')
    raise RuntimeError

def print_ok(*message):
    print('\033[92m', *message, '\033[0m')

def print_warning(*message):
    print('\033[93m', *message, '\033[0m')

def print_info(*message):
    print('\033[96m', *message, '\033[0m')

def print_white(*message):
    print('\033[37m', *message, '\033[0m')

def pathlib_file(file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    elif not isinstance(file_name, Path):
        raise TypeError(f'Please check the type of '
                        f'the filename:{file_name}')
    return file_name

def save_to_json(data, file_name):
    file_name = pathlib_file(file_name)
    if not file_name.parent.exists():
        Path.mkdir(file_name.parent, parents=True)
    with file_name.open('w') as f:
        json.dump(data, f, indent=2)

def load_from_json(file_name):
    file_name = pathlib_file(file_name)
    with file_name.open('r') as f:
        data = json.load(f)
    return data

def plot_loss(losses, labels, title=None, font_size=20, subsample=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='x', labelsize=font_size)
    ax.xaxis.get_offset_text().set_fontsize(font_size)
    ax.yaxis.get_offset_text().set_fontsize(font_size)
    ax.xaxis.set_tick_params(labelsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)
    for loss, label in zip(losses, labels):
        if len(loss.shape) > 1:
            x = loss[:, 0]
            y = loss[:, 1]
        else:
            x = np.arange(len(loss))
            y = loss
        if subsample is not None:
            x = x[::subsample]
            y = y[::subsample]
        ax.plot(x, y, label=label)
    ax.set_xlabel('No. of iterations', fontsize=font_size)
    ax.set_ylabel('Loss', fontsize=font_size)
    if title is not None:
        ax.set_title(title, fontsize=font_size)
    ax.legend(fontsize=font_size)
    return fig, ax
