import tensorflow as tf
x = tf.constant([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
print(x.shape)
xt = tf.transpose(x, perm = [2,1,0])

with tf.Session() as sess:
    print(x.eval(), "\n")
    print(xt.eval(), "\n", "transposed shape ", xt.shape)
    sess.close()