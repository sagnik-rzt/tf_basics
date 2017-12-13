import tensorflow as tf

def scalar_multiplication(a, x):
    return tf.scalar_mul(scalar = a, x = x)

a = tf.placeholder(dtype= tf.float32)
b = tf.constant(value = 5, dtype = tf.float32)
c = scalar_multiplication(b, a)

def matrix_multiply(x1, x2):
    return tf.matmul(x1, x2)

A = tf.ones(shape = [3,1], dtype= tf.float32)
B = tf.ones(shape = [1,3], dtype= tf.float32)
C = matrix_multiply(A, B)

with tf.Session() as sess:
    result1 = sess.run(c, feed_dict = {a : [1,2,3]})
    result2 = sess.run(C)
    print(result1, '\n')
    print(result2)
