import tensorflow as tf
import numpy as np

const = tf.constant(2.0, name='const')
# b is now an array of values (float32's)
b = tf.placeholder(tf.float32, [None, 1], name='b')
# c is also an array of values (floats32's)
c = tf.placeholder(tf.float32, [None, 1], name='c')

## let's define operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

init_op = tf.global_variables_initializer()

## b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
## a_out is [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
with tf.Session() as sess:
    # initialize the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a,
                     feed_dict={b: np.arange(0, 10)[:, np.newaxis],
                                c: np.arange(0, 10)[:, np.newaxis]
                     })
    print('Variable a is {}'.format(a_out))
