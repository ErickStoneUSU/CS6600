import gym
from hw06.Model import Model
from hw06.Memory import Memory

import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import random
import math

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 50

class MCar:
    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0
        max_x = -100
        while True:
            if self._render:
                self._env.render()
                
            action = self._choose_action(state)
            # a call to the environment self._env.step(action) returns four elements:
            # next_state - this is a 1D numpy array (numpy.ndarray), e.g., [-0.531827   -0.00216068];
            # - next_state[0] is the car's position; the position is a float in [-1.2, 0.6]
            # - next_state[1] is the car's velocity; the velocity is a float in [-0.07, 0.07]
            # reward is a float, e.g., -1.0;
            # done is a boolean;
            # info is a dictionary.
            next_state, reward, done, info = self._env.step(action)
            # this is where we can modify the reward returned by the environment.
            # if the car's position is at or above 0.1, we increase the reward by 10.
            if next_state[0] >= 0.1:
                reward += 10
            # if the car's position is at or above 0.25, we increase the reward by 20.
            elif next_state[0] >= 0.25:
                reward += 20
            # if the car's position is at or above 0.5, we increase the reward by 100.
            elif next_state[0] >= 0.5:
                reward += 100

            # keep track of the righmost x position ever reached by the car.
            if next_state[0] > max_x:
                max_x = next_state[0]
            # is the game complete? If so, set the next state to None.
            if done:
                next_state = None

            # we commit the (state, action, reward, next_state) to memory
            self._memory.add_sample((state, action, reward, next_state))
            # and replay. replay is where RL actually happens.
            self._replay()

            # exponentially decay the epsilon value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) \
                        * math.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._reward_store.append(tot_reward)
                self._max_x_store.append(max_x)
                break

            print('Step {}, Total reward: {}, Eps: {}'.format(self._steps,
                                                              tot_reward, self._eps))

    # play is run w/o any learning.
    def play(self):
        state = self._env.reset()
        tot_reward = 0
        max_x = -100
        self._reward_store = []
        self._max_x_store = []
        while True:
            if self._render:
                self._env.render()
                
            action = self._play_choose_action(state)

            next_state, reward, done, info = self._env.step(action)

            # keep track of the righmost x position ever reached by the car.
            if next_state[0] > max_x:
                max_x = next_state[0]
            # is the game complete? If so, set the next state to None.
            if done:
                next_state = None

            # exponentially decay the eps value
            self._steps += 1
            #self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) \
            #            * math.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._reward_store.append(tot_reward)
                self._max_x_store.append(max_x)
                print('Step {}, Total reward: {}'.format(self._steps, tot_reward))
                break

            print('Step {}, Total reward: {}'.format(self._steps, tot_reward))

    # choose a random number in [0, 1]; if the number is < than _eps,
    # randomly select an action; else choose the best action predicted
    # by the model.
    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    # choose the action according to a learned model.
    def _play_choose_action(self, state):
        return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        # batch is list of 4-toops of the form
        # (array([-0.45607093, -0.00334033]), 1, -1.0, array([-0.45991426, -0.00384333])),
        # where the 1st element is a state, the second is action, the third is reward,
        # and the fourth is next_state.
        batch = self._memory.sample(self._model.batch_size)
        #print('len of batch = ' + str(len(batch)))
        # collect all states in batch. states will be an array of 2-toops:
        # states=[[-5.20062426e-01 -2.31394437e-03],
        #         [-5.58174462e-01 -4.54577626e-03],
        #         [-5.17748481e-01 -2.27006942e-03],
        #         ...]]
        states = np.array([val[0] for val in batch])
        # collection next_states. if the next state is None, the collected next_state is [0, 0];
        # next_states is an array of 2-toops of the same form as states above.
        next_states = np.array([(np.zeros(self._model.num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states;
        # q_s_a will be an ? x 3 numpy array, where each row is a 3-toop of the form
        # [-20.580538, -21.274868, -21.296692], each of the three numbers denote
        # a reward for taking a specific action (e.g., 0 - push left, 1 - no push, 2 - push right)
        # in a specific state, i.e., a 2-toop whose 0th element is position and whose 1st element
        # is velocity.
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        # q_s_a_d has the same structure as q_s_a but it is predicted from next_states.
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        # x is batch x 2
        x = np.zeros((len(batch), self._model.num_states))
        # y is batch x 3
        y = np.zeros((len(batch), self._model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # current_q is of the form [-50.88421, -51.148014, -50.96282], where the 0th value
            # is the reward for action 0, the 1st value is the reward for action 1, and the
            # 2nd value is the reward for action 2.
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                # this is the Q reward function. The reward for the action taken in
                # current_q is the same as Q(s, a), where action is the action that
                # took the agent from the current state to the next state.
                # np.amax(q_s_a_d[i]) is the maximum reward in the next state
                # So, the update rule is Q(s, a) = r + gamma * max(q_s_a_d[i]).
                # But, where is alpha?
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        # train_batch uses AdamOptimizer to minimize results between x and y
        # x's are states and y's are 3-toop rewards.
        self._model.train_batch(self._sess, x, y)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._max_x_store

    @property
    def model(self):
        return self._model

    @property
    def steps(self):
        return self._steps

def train_mcar():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    num_states = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.n

    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(50000)

    with tf.Session() as sess:
        sess.run(model.var_init)
        mc = MCar(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)
        # change the number of episodes as needed
        num_episodes = 10
        cnt = 0
        while cnt < num_episodes:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
            mc.run()
            cnt += 1
            
        plt.plot(mc.reward_store)
        plt.show()
        plt.close("all")
        plt.plot(mc.max_x_store)
        plt.show()

if __name__ == '__main__':
    train_mcar()
    
