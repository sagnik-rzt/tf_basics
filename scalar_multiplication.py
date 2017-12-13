import tensorflow as tf

def scalar_multiplication(a, x):
    return tf.scalar_mul(scalar = a, x = x)

a = tf.placeholder(dtype= tf.float32)
b = tf.constant(value = 5, dtype = tf.float32)
c = scalar_multiplication(b, a)

with tf.Session() as sess:
    result = sess.run(c, feed_dict = {a : [1,2,3]})
    print(result)