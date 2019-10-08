import tensorflow as tf

# let's define variables and constants
CONST_1 = tf.constant(1.0, name='CONST_1')
A = tf.Variable(5.0, name='A')
B = tf.Variable(3.0, name='B')

## let's define operations and outputs.
C = tf.add(A, B, name='C')
D = tf.add(B, CONST_1, name='D')
E = tf.multiply(C, D, name='E')

## let's define the tensorflow initialization operation.
init_op = tf.global_variables_initializer()

## define a tensorflow session and run it.
with tf.Session() as sess:
    # initialize the variables
    sess.run(init_op)
    # compute the output of the graph
    EV = sess.run(E)
    print('Variable E = {}'.format(EV))
