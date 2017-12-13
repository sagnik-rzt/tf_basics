import tensorflow as tf

def find_max_argument(x, axis_index):
    return tf.argmax(x, axis = axis_index)

def find_min_argument(x, axis_index):
    return tf.argmin(x, axis = axis_index)

a = tf.constant([[1,2,3],[4,5,6]], dtype = tf.float32)
b = find_max_argument(a, axis_index = 0)
c = find_max_argument(a, axis_index = 1)
d = find_min_argument(a, axis_index = 0)
e = find_min_argument(a, axis_index = 1)

with tf.Session() as sess:
    result1 = sess.run(b)
    result2 = sess.run(c)
    result3 = sess.run(d)
    result4 = sess.run(e)
    print(result1, '\n')
    print(result2, '\n')
    print(result3, '\n')
    print(result4)