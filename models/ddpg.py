from collections import deque
from tflearn.initializations import uniform
from tflearn.layers.core import *
from tflearn.layers.conv import *
from utils import PolicyNetwork, Buffer
import tflearn as tl
import tensorflow as tf

# TODO: Should get soylent?
class DeepDeterministicPG(PolicyNetwork):
    def __init__(self, channels=1, hidden_size=512, learning_rate=0.001,\
        image_size_x=80, image_size_y=80, discount_factor=0.99, max_memory=10000,\
        logging=True):
        # Network parameters
        self.channels = channels
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.image_size_x, self.image_size_y = image_size_x, image_size_y
        self.discount_factor = discount_factor
        self.max_memory = max_memory
        self.logging = logging
        
        # Network I/O streams
        self.input_shape = [None, self.image_size_x * self.image_size_y * self.channels]
        self.output_shape = [None, 1]
        self.x = tf.placeholder(shape=self.input_shape, dtype=tf.float32)
        self.y = tf.placeholder(shape=self.output_shape, dtype=tf.int32)
        self.discounted_rewards = tf.placeholder(shape=self.output_shape, dtype=tf.float32)
        self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32)

        # Initialize various networks
        self.actor = self._actor_graph()
        self.critic = self._critic_graph()
        
        # Memory initialization
        self._init_memory()

    def _init_memory(self):
        self._buffer = Buffer(max_size=max_memory)

    def _actor_graph(self):
        with tf.name_scope('actor_network'):
            inputs = tf.reshape(self.x, [-1, self.image_size_x, self.image_size_y])
            
    def _critic_graph(self):
        with tf.name_scope('critic_network'):

def get_policy_network():
    return DeepDeterministicPG