import tensorflow as tf
import tflearn as tl
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tqdm import tqdm
from numpy.random import uniform
from numpy import exp

FLAGS = tf.app.flags.FLAGS


class PolicyNetwork(object):
    def __init__(self):
        # Network parameters
        self.channels = FLAGS.channels
        self.hidden_size = FLAGS.hidden_size
        self.learning_rate = FLAGS.learning_rate
        self.image_size_x, self.image_size_y = FLAGS.image_size_x, FLAGS.image_size_y
        self.discount_factor = FLAGS.discount_factor

        # Network I/O streams
        self.input_shape = [None, self.image_size_y * self.image_size_x * self.channels]
        self.output_shape = (None,)
        self.x = tf.placeholder(shape=self.input_shape, dtype=tf.float32, name='input')
        self.y = tf.placeholder(shape=self.output_shape, dtype=tf.float32, name='taken_actions')
        self.discounted_rewards = tf.placeholder(shape=self.output_shape, dtype=tf.float32, name='rewards')
        self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32, name='dropout_prob')

        # Initialize network and optimizer for training
        self.logits = self._inference_graph()
        self.network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,\
                                              scope='policy_network')
        self.loss, self.gradients = self._loss()
        self.grads = [tf.placeholder(shape=gradient.get_shape(), dtype=tf.float32)\
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
        self.grad_buffer = [0.0 * self.sess.run(grad) for grad in self.network_vars]

    def _reset_memory(self, reset_gradient=False):
        self.states = []
        self.actions = []
        self.rewards = []

        if reset_gradient:
            self.grad_buffer = [0.0 * self.sess.run(grad) for grad in self.network_vars]

    def _inference_graph(self):
        with tf.name_scope('policy_network'):
            reshaped = tf.reshape(self.x, [-1, self.image_size_x, self.image_size_y, self.channels])
            conv_1 = conv_2d(reshaped, 32, 3, activation='relu', name='conv_1')
            max_pool_1 = max_pool_2d(conv_1, 2, name='max_pool_1')
            conv_2 = conv_2d(max_pool_1, 64, 3, activation='relu', name='conv_2')
            max_pool_2 = max_pool_2d(conv_2, 2, name='max_pool_3')
            fc_1 = fully_connected(max_pool_2, self.hidden_size, activation='relu', name='fc_1')
            drop = dropout(fc_1, keep_prob=self.keep_prob, name='dropout')
            fc_2 = fully_connected(drop, 1, activation='relu', name='fc_2')
            out = tf.nn.sigmoid(fc_2, name='sigmoid_out')

            return out

    def _loss(self):
        pg_loss = tf.reduce_mean(
            tf.square(self.y - self.logits) * self.discounted_rewards
        )

        if FLAGS.logging:
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
                self.y: self.actions,
                self.discounted_rewards: drs,
                self.keep_prob: 1.0 
            }
        )

        # Updated gradient buffer
        for i, gradient in enumerate(gradients):
            self.grad_buffer[i] += gradient

        if train_batch:
            print "Training batch of size %d..." % len(self.rewards)
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
        
        action = 2 if prob[0][0][0] < uniform() else 3

        return action

    def update_memory(self, state, action, reward, gradients=None):
        assert len(self.states) == len(self.actions) and len(self.actions) == len(self.rewards)

        self.states.append(state)
        self.actions.append(0.0 if action == 2 else 1.0)
        self.rewards.append(reward)