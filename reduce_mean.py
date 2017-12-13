import tensorflow as tf

def find_reduced_mean(x, axis_index):
    return tf.reduce_mean(x, axis = axis_index)

a = tf.ones(shape = [3,4], dtype = tf.float32)
b = find_reduced_mean(a, axis_index = 0)
c = find_reduced_mean(a, axis_index = 1)

with tf.Session() as sess:
    result1 = sess.run(b)
    result2 = sess.run(c)
    print(result1, '\n')
    print(result2)