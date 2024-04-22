from matplotlib import pyplot as plt
import tensorflow as tf
from loadData import mappable

data = tf.data.Dataset.list_files('./data/s1/*.mpg').shuffle(500).map(mappable).padded_batch(2, padded_shapes=([75, None, None, None], [40])).prefetch(tf.data.AUTOTUNE)

frames, alignments = data.as_numpy_iterator().next()

print(frames)