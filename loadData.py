from loadAlignments import load_alignments
from videoLoading import load_video
import tensorflow as tf
import os


def load_data(path):
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments

def mappable(path):
    return tf.py_function(load_data, [path], (tf.float32, tf.int64))
