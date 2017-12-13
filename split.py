import tensorflow as tf

def split_tensor(x, split_sequence, axis_index):
    return tf.split(x, num_or_size_splits = split_sequence, axis = axis_index)

a = tf.ones(shape = [3,3], dtype = tf.float32)
b = split_tensor(a, split_sequence = [2,1], axis_index = 0)

with tf.Session() as sess:
    result = sess.run(b)
    print(result)