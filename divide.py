import tensorflow as tf

def divide(x1, x2):
    return tf.divide(x1, x2)

a = tf.placeholder(dtype = tf.float32)
b = tf.placeholder(dtype = tf.float32)
c = divide(a, b)

with tf.Session() as sess:
    result = sess.run(c, feed_dict = {a : [10,20,30], b : [5,10,15]})
    print(result)