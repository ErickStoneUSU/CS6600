import tensorflow as tf

## In the mountain car game, the number of states is 2 (position, velocity) and
## num of actions is 3 (push left, no push, push right).
from tflearn import DNN

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations       self._logits = None # this is the output of an ANN.
        self._optimizer = None
        self._var_init = None
        self._fc1 = None
        self._fc2 = None
        self._fc3 = None
        self.num = 0
        # now setup the model
        self._define_model()
        
    def _define_model(self):
        self._define_model_5()

    def _define_model_1(self):
        self.num = 1
        self._states = tf.placeholder(shape=[None, self._num_states],
                                      dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions],
                                     dtype=tf.float32)
        self._fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        self._fc2 = tf.layers.dense(self._fc1, 50, activation=tf.nn.relu)
        self._fc3 = tf.layers.dense(self._fc2, 50, activation=tf.nn.relu)
        self._r1 = tf.math.round(self._fc3, 2)
        self._logits = tf.layers.dense(self._r1, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()


    def _define_model_2(self):
        self.num = 2
        self._states = tf.placeholder(shape=[None, self._num_states],
                                      dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions],
                                     dtype=tf.float32)
        self._fc1 = tf.layers.dense(self._states, 512, activation=tf.nn.relu)
        self._fc2 = tf.layers.dense(self._fc1, 512, activation=tf.nn.relu)
        self._fc3 = tf.layers.dense(self._fc2, 512, activation=tf.nn.relu)
        self._logits = tf.layers.dense(self._fc3, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def _define_model_3(self):
        self.num = 3
        self._states = tf.placeholder(shape=[None, self._num_states],
                                      dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions],
                                     dtype=tf.float32)
        self._fc1 = tf.layers.dense(self._states, 256, activation=tf.nn.relu)
        self._fc2 = tf.layers.dense(self._fc1, 512, activation=tf.nn.relu)
        self._fc3 = tf.layers.dense(self._fc2, 1024, activation=tf.nn.relu)
        self._logits = tf.layers.dense(self._fc3, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def _define_model_4(self):
        self.num = 4
        self._states = tf.placeholder(shape=[None, self._num_states],
                                      dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions],
                                     dtype=tf.float32)
        self._fc1 = tf.layers.dense(self._states, 1024, activation=tf.nn.relu)
        self._fc2 = tf.layers.dense(self._fc1, 512, activation=tf.nn.relu)
        self._fc3 = tf.layers.dense(self._fc2, 256, activation=tf.nn.relu)
        self._logits = tf.layers.dense(self._fc3, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def _define_model_5(self):
        self.num = 5
        self._states = tf.placeholder(shape=[None, self._num_states],
                                      dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions],
                                     dtype=tf.float32)
        self._fc1 = tf.layers.dense(self._states, 512, activation=tf.nn.relu)
        self._fc2 = tf.layers.dense(self._fc1, 256, activation=tf.nn.relu)
        self._fc3 = tf.layers.dense(self._fc2, 512, activation=tf.nn.relu)
        self._logits = tf.layers.dense(self._fc3, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    # take a state and a session and use the network to predict
    # the next state. state is a vector of 2 floats, e.g., [-0.61506952 -0.00476815].
    def predict_one(self, state, sess):
        #print('predict_one: state={}'.format(state))
        next_state = sess.run(self._logits, feed_dict={self._states:
                                                       state.reshape(1,
                                                                     self.num_states)})
        #print('predict_one: next_state={}'.format(next_state))
        # next_state is a vector state predicted by the network (e.g., [[-53.998287 -53.755825 -53.867805]])
        return next_state
    
    # sess is tf.Session.
    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

    def set_fc1(self, fc1):
        self._fc1 = fc1

    def set_fc2(self, fc2):
        self._fc2 = fc2

    def set_logits(self, logits):
        self._logits = logits

    @property
    def fc1(self):
        return self._fc1

    @property
    def fc2(self):
        return self._fc2

    @property
    def logits(self):
        return self._logits

    @property
    def b(self):
        return self._b

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init
