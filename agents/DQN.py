"""
Taken from https://github.com/MorvanZhou/ with minor changes in layer layout and initializations
"""
import random
from collections import deque

import numpy as np
import tensorflow as tf
import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch.optim import Adam
from typing import NamedTuple, Tuple
import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork:
    _memory_counter = 0
    _epsilon = 0.0
    _learning_step_counter = 0

    def __init__(self,
                 dim_actions: int,
                 dim_states: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 replace_target_iter: int = 250,
                 memory_size: int = 10000,
                 batch_size: int = 500,
                 epsilon_max: float = 0.9,
                 epsilon_incr: float = 0.0001):
        # assign variables
        self._dim_actions = dim_actions
        self._dim_states = dim_states
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._replace_target_iter = replace_target_iter
        self._memory_size = memory_size
        self._batch_size = batch_size
        self._epsilon_incr = epsilon_incr
        self._epsilon_max = epsilon_max

        # initialize memory [s, a, r, s_]
        self._memory = np.zeros((self._memory_size, dim_states * 2 + 2))

        # build net
        self._build_network()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self._target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self._sess = tf.Session()

        tf.summary.scalar('loss', self._loss)
        self._merge_summary_op = tf.summary.merge_all()
        self._summary_writer = tf.summary.FileWriter("logs/run5", self._sess.graph)

        self._sess.run(tf.global_variables_initializer())

    def _build_network(self):
        self.s = tf.placeholder(tf.float32, [None, self._dim_states], name='state')
        self.s_ = tf.placeholder(tf.float32, [None, self._dim_states], name='next_state')
        self.r = tf.placeholder(tf.float32, [None, ], name='reward')
        self.a = tf.placeholder(tf.int32, [None, ], name='action')

        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.0)

        # build evaluation network
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 24, tf.nn.tanh, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 48, tf.nn.tanh, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            self.q_eval = tf.layers.dense(e2, self._dim_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # build target network
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 24, tf.nn.tanh, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 48, tf.nn.tanh, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            self.q_next = tf.layers.dense(t2, self._dim_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_next')

        with tf.variable_scope('q_target'):
            q_target = self.r + self._gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)  # don't want to backprop the target Q network, so stop_gradient

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        with tf.variable_scope('loss'):
            # we update eval Q network constantly, by taking the difference from the target Q network
            self._loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a), name='TD_error')

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace old memory with new memory
        index = self._memory_counter % self._memory_size
        self._memory[index, :] = transition
        self._memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when fed into tensorflow
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self._epsilon:
            # forward feed the observation to get Q value for every action
            action_values = self._sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self._dim_actions)
        return action

    def learn(self):
        # check to replace target params
        if self._learning_step_counter % self._replace_target_iter == 0:
            self._sess.run(self._target_replace_op)
            print(f"\ntarget_params_replaced, {self._epsilon}\n")

        # sample batch memory from all memory
        if self._memory_counter > self._memory_size:
            sample_index = np.random.choice(self._memory_size, size=self._batch_size)
        else:
            sample_index = []
        batch_memory = self._memory[sample_index, :]

        _, _, summary = self._sess.run(
            [self._train_op, self._loss, self._merge_summary_op],
            feed_dict={
                self.s: batch_memory[:, :self._dim_states],
                self.a: batch_memory[:, self._dim_states],
                self.r: batch_memory[:, self._dim_states + 1],
                self.s_: batch_memory[:, -self._dim_states:],
            })
        self._summary_writer.add_summary(summary, self._learning_step_counter)

        # increase epsilon ?
        self._epsilon = min(self._epsilon + self._epsilon_incr, self._epsilon_max)

        self._learning_step_counter += 1


class ALEDeepQNetwork(DeepQNetwork):
    def _build_network(self):
        self.s = tf.placeholder(tf.uint8, [None, 84, 84, 4], name='state')  # screen
        self.s_ = tf.placeholder(tf.float32, [None, 84, 84, 4], name='next_state')  # next screen
        self.r = tf.placeholder(tf.float32, [None, ], name='reward')
        self.a = tf.placeholder(tf.int32, [None, ], name='action')

        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.0)

        # build evaluation network
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.conv2d(self.x, filters=32, kernel_size=(8, 8), strides=4, activation=tf.nn.relu,
                                  kernel_initializer=w_initializer, bias_initializer=b_initializer)
            e2 = tf.layers.conv2d(e1, filters=64, kernel_size=(4, 4), strides=2, activation=tf.nn.relu,
                                  kernel_initializer=w_initializer, bias_initializer=b_initializer)
            e3 = tf.layers.conv2d(e2, filters=64, kernel_size=(3, 3), strides=1, activation=tf.nn.relu,
                                  kernel_initializer=w_initializer, bias_initializer=b_initializer)
            e1 = tf.layers.dense(self.s, 24, tf.nn.tanh, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 48, tf.nn.tanh, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            self.q_eval = tf.layers.dense(e2, self._dim_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # build target network
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 24, tf.nn.tanh, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 48, tf.nn.tanh, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            self.q_next = tf.layers.dense(t2, self._dim_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_next')

        with tf.variable_scope('q_target'):
            q_target = self.r + self._gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)  # don't want to backprop the target Q network, so stop_gradient

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        with tf.variable_scope('loss'):
            # we update eval Q network constantly, by taking the difference from the target Q network
            self._loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a), name='TD_error')

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)


class Transition(NamedTuple):
    state: Tuple[float, ...]
    action: int
    reward: float
    next_state: Tuple[float, ...]


class ExperienceReplay:
    def __init__(self, memory_size):
        self._memory = deque(maxlen=memory_size)

    def add(self, transition: Transition):
        self._memory.append(transition)

    def sample(self, batch_size) -> Transition:
        return random.sample(self._memory, min(len(self._memory), batch_size))


class Network(nn.Module):
    def __init__(self, dim_states: int, dim_actions: int):
        super(Network, self).__init__()
        self.input = nn.Linear(dim_states, 24)
        self.hidden1 = nn.Linear(24, 48)
        self.output = nn.Linear(48, dim_actions)

    def forward(self, x):
        x = F.relu(self.hidden(self.input(x)))
        return self.output(x.view(x.size(0), -1))


class DQN:
    _episode_count = 0
    _epsilon = 0.

    def __init__(self,
                 dim_actions: int,
                 dim_states: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 replace_target_iter: int = 250,
                 memory_size: int = 10000,
                 batch_size: int = 500,
                 epsilon_max: float = 0.9,
                 epsilon_incr: float = 0.0001):
        # assign variables
        self._dim_actions = dim_actions
        self._dim_states = dim_states
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._update_target_network = replace_target_iter
        self._batch_size = batch_size
        self._epsilon_max = epsilon_max
        self._epsilon_incr = epsilon_incr

        # initialize required stores
        self._memory = ExperienceReplay(memory_size)
        self._policy_network = Network(dim_states, dim_actions)
        self._target_network = Network(dim_states, dim_actions)
        self._optimizer = Adam(self._policy_network.parameters())

        # copy eval_network to target_network
        self._target_network.load_state_dict(self._policy_network.state_dict())

    def store_transition(self, s, a, r, s_):
        transition = Transition(state=s, action=a, reward=r, next_state=s_)
        self._memory.add(transition)

    def choose_action(self, observation):
        print('choose_action')
        if np.random.uniform() < self._epsilon:
            # forward feed the obseration to geth the Q value for state
            return self._policy_network(Variable(observation, volatile=True).type(FloatTensor)).max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(self._dim_actions)]])

    def learn(self):
        self._update_target()
        batch = self._memory.sample(self._batch_size)

        # get the expected rewards for each state in batch
        state_batch = torch.cat([transition.state for transition in batch])
        state_action_values = self._policy_network(state)


    def _update_target(self):
        if self._episode_count % self._update_target_network == 0:
            self._target_network.load_state_dict(self._policy_network.state_dict())
