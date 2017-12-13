import tensorflow as tf

def concatenate_tensors(x1, x2, axis_index):
    return tf.concat(values = [x1, x2], axis = axis_index)

a = tf.ones(shape = [3,3], dtype = tf.float32)
b = tf.zeros(shape = [3,3], dtype = tf.float32)
c = concatenate_tensors(a, b, axis_index= 0)
d = concatenate_tensors(a, b, axis_index= 1)

with tf.Session() as sess:
    result1 = sess.run(c)
    result2 = sess.run(d)
    print(result1, '\n')
    print(result2, '\n')
