"""
Taken from https://github.com/MorvanZhou/ with minor changes in layer layout and initializations
"""
import tensorflow as tf
import numpy as np


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
                 epsilon_incr: float=0.0001):
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
