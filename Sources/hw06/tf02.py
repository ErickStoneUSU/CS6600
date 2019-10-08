import tensorflow as tf
import numpy as np

const = tf.constant(2.0, name='const')
# b is now an array of values (float32's)
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, name='c')

## let's define a few tensorflow operations.
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

## let's define the tensor flow initialization operation.
init_op = tf.global_variables_initializer()

## here is how you can construct a feed dictionary
## to a tensor. A feed dictionary is a dictionary
## of place holders and their values that are feed to
## a given tensor.
## >>> d = {b: np.arange(0, 5)[:,np.newaxis]}
## >>> d
## {<tf.Tensor 'b:0' shape=(?, 1) dtype=float32>: array([[0],
##       [1],
##       [2],
##       [3],
##       [4]])}

## b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
## then a_out is [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
with tf.Session() as sess:
    # initialize the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a,
                     feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print('Variable a is {}'.format(a_out))
