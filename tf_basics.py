import tensorflow as tf


def add_tensors(x1, x2):
    return tf.add(x1, x2)

def subtract_tensors(x1, x2):
    return tf.subtract(x1, x2)

def elementwise_multiply(x1, x2):
    return tf.multiply(x1, x2)

def scalar_multiply(a, x1):
    return tf.scalar_mul(a, x1)

def matrix_multiply(x1, x2):
    return tf.matmul(x1, x2)

def split_tensor(x1, splitting_sequence, axis_index):
    return tf.split(x1, num_or_size_splits= splitting_sequence, axis = axis_index)

def reshape_tensor(x1, final_shape):
    return tf.reshape(x1, shape = final_shape)

def find_transpose(x1, permutation):
    return tf.transpose(x1, perm = permutation)

def concatenate_tensors(x1, x2, axis_index):
    return tf.concat([x1, x2], axis = axis_index)

def find_maximum_argument(x1, axis_index):
    return tf.argmax(x1, axis = axis_index)

def find_minimum_argument(x1, axis_index):
    return tf.argmin(x1, axis = axis_index)

def find_reduced_mean(x1, axis_index):
    return tf.reduce_mean(x1, axis = axis_index)


def main():

    a = tf.placeholder(dtype = tf.float32)
    b = tf.placeholder(dtype = tf.float32)
    c = tf.constant(value = 5.0, dtype = tf.float32)
    d = tf.Variable(initial_value = [1.0,2.0,3.0], dtype = tf.float32)

    z = scalar_multiply(c, add_tensors(elementwise_multiply(a, b), d))
    z1, z2= split_tensor(z, splitting_sequence= [1,2], axis_index = 0)
    tf.add_to_collection(name = "var1", value = z)

    saver1 = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        result1 = sess.run(z, feed_dict = {a : [10.0,20.0,30.0], b : [-5.0, -6.0, -7.0]})
        print(result1)
        result2 = sess.run([z1, z2], feed_dict = {a : [10.0,20.0,30.0], b : [-5.0, -6.0, -7.0]})
        print(result2)
        saver1.save(sess, save_path = "saver1/test.model")


main()