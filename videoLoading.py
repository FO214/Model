import cv2
import tensorflow as tf

def load_video(path):
    c = cv2.VideoCapture(path)
    frames = []

    for i in range(int(c.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = c.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])

    c.release()

    return tf.cast((frames - tf.math.reduce_mean(frames)), tf.float32) / tf.math.reduce_std(tf.cast(frames, tf.float32))
