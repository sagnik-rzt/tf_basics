import tensorflow as tf

def subtract_tensors(x1, x2):
    return tf.subtract(x1, x2)

a = tf.placeholder(dtype= tf.float32)
b = tf.placeholder(dtype= tf.float32)
c = subtract_tensors(a, b)

with tf.Session() as sess:
    result = sess.run(c, feed_dict = {a : [1,2,3], b : [4,5,6]})
    print(result)