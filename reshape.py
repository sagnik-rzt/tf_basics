import tensorflow as tf

def reshape_tensor(x, final_shape):
    return tf.reshape(x, shape = final_shape)

a = tf.ones(shape = [2,3], dtype = tf.float32)
b = reshape_tensor(a, final_shape = [3,2])
c = reshape_tensor(a, final_shape = [-1])

with tf.Session() as sess:
    result1 = sess.run(b)
    result2 = sess.run(c)
    print(result1, '\n')
    print(result2)
