import tensorflow as tf

def matrix_multiply(x1, x2):
    return tf.matmul(x1, x2)

a = tf.ones(shape = [3,1], dtype= tf.float32)
b = tf.ones(shape = [1,3], dtype= tf.float32)
c = matrix_multiply(a, b)

with tf.Session() as sess:
    result = sess.run(c)
    print(result)