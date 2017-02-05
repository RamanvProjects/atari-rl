from tflearn.layers.core import *
from tflearn.layers.conv import *
from utils import PolicyNetwork
from tqdm import tqdm
from numpy.random import uniform
from numpy import exp
import tensorflow as tf
import tflearn as tl


class ReinforcePG(PolicyNetwork):
    def __init__(self, channels=1, hidden_size=512, learning_rate=0.001,\
        image_size_x=80, image_size_y=80, discount_factor=0.99, max_memory=10000,\
        num_actions=6, logging=True):
        # Network parameters
        self.channels = channels
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.image_size_x, self.image_size_y = image_size_x, image_size_y
        self.discount_factor = discount_factor
        self.max_memory = max_memory
        self.logging = logging
        self.num_actions = num_actions

        # Network I/O streams
        self.input_shape = (None, self.image_size_y * self.image_size_x * self.channels)
        self.output_shape = (None, 1)
        self.x = tf.placeholder(shape=self.input_shape, dtype=tf.float32, name='input')
        self.y = tf.placeholder(shape=self.output_shape, dtype=tf.int32, name='taken_actions')
        self.discounted_rewards = tf.placeholder(shape=self.output_shape, dtype=tf.float32, name='rewards')
        self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32, name='dropout_prob')

        # Initialize network and optimizer for training
        self.logits = self._inference_graph()
        self.network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='policy_network')
        self.loss, self.gradients = self._loss()
        self.grads = [tf.placeholder(shape=gradient.get_shape(), dtype=tf.float32)
                      for gradient in self.network_vars]
        self.optimizer = self._optimizer()

        # Initialize tf session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        # Initialize memory storage for rollouts
        self._init_memory()
    
    def _init_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.num_examples_seen = 0
        self.grad_buffer = [0.0 * self.sess.run(grad) for grad in self.network_vars]

    def _reset_memory(self, reset_gradient=False):
        self.states = []
        self.actions = []
        self.rewards = []
 
        if reset_gradient:
            self.grad_buffer = [0.0 * self.sess.run(grad) for grad in self.network_vars]
            self.num_examples_seen = 0

    def _inference_graph(self):
        with tf.name_scope('policy_network'):
            reshaped = tf.reshape(self.x, [-1, self.image_size_x, self.image_size_y, self.channels])
            conv_1 = conv_2d(reshaped, 32, 3, activation='relu', name='conv_1')
            max_pool_1 = max_pool_2d(conv_1, 2, name='max_pool_1')
            conv_2 = conv_2d(max_pool_1, 64, 3, activation='relu', name='conv_2')
            max_pool_2 = max_pool_2d(conv_2, 2, name='max_pool_3')
            fc_1 = fully_connected(max_pool_2, self.hidden_size, activation='relu', name='fc_1')
            drop = dropout(fc_1, keep_prob=self.keep_prob, name='dropout')
            fc_2 = fully_connected(drop, self.num_actions, activation='relu', name='fc_2')
            out = tf.nn.softmax(fc_2, name='softmax_out')

            return out

    def _loss(self):
        # TODO: Switch loss to use logarithm
        one_hot_y = tf.one_hot(self.y, self.num_actions, 1.0, 0.0, axis=-1)
        repeated_rewards = tf.tile(self.discounted_rewards, [1, self.num_actions])
        pg_loss = tf.reduce_mean(
            -tf.reduce_sum(tf.square(one_hot_y - self.logits) * repeated_rewards)
        )

        if self.logging:
            tf.summary.scalar('Loss', pg_loss)
        
        gradients = tf.gradients(pg_loss, self.network_vars)
        return pg_loss, gradients
    
    def _optimizer(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        return opt.apply_gradients(zip(self.grads, self.network_vars))

    def partial_fit_step(self, train_batch=False):
        """
        Train network on rewards. Uses state and action memory with these rewards
        returns loss
        """
        # Discount rewards
        N = len(self.rewards)
        self.num_examples_seen += N
        r = 0

        drs = np.zeros(N)
        for t in reversed(xrange(N)):
            r = self.rewards[t] + r * self.discount_factor
            drs[t] = r

        # Normalize rewards
        drs -= np.mean(drs)
        drs /= np.std(drs)

        loss, gradients = self.sess.run(
            [self.loss, self.gradients],
            feed_dict={
                self.x: self.states,
                self.y: np.vstack(self.actions),
                self.discounted_rewards: np.vstack(drs),
                self.keep_prob: 1.0 
            }
        )

        # Updated gradient buffer, we use a buffer over several episodes
        # since it's usually very small as the reward signal is sparse
        for i, gradient in enumerate(gradients):
            self.grad_buffer[i] += gradient

        if train_batch:
            print "***Training*** batch of size %d..." % self.num_examples_seen
            feed_grad = dict(zip(self.grads, self.grad_buffer))
            self.sess.run(self.optimizer, feed_dict=feed_grad)

        rewards = self.rewards
        self._reset_memory(reset_gradient=train_batch)
       
        return loss, rewards

    def get_action(self, X):
        """
        Returns action based on network probability of going up based on an input image
        """
        if len(X.shape) != 2:
            length = X.shape[0]
            X = X.reshape([1, length])

        prob = self.sess.run([self.logits],
                             feed_dict={
                                 self.x: X,
                                 self.keep_prob: 1.0
                             })
        action = int(np.random.choice(self.num_actions, 1, p=prob[0][0])[0])
        return action

    def update_memory(self, state, action, reward, t, next_state=None):
        assert len(self.states) == len(self.actions) and len(self.actions) == len(self.rewards)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

        if len(self.states) > self.max_memory:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)

def get_policy_network():
    return ReinforcePG