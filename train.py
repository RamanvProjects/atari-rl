from models import policy
import tensorflow as tf
import gym
import utils


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 10, 'Number of episodes per batch')
flags.DEFINE_integer('epoch_size', 100, 'Number of batches per epoch')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for network')
flags.DEFINE_integer('epochs', 15, 'Number of epochs, -1 for infinite')
flags.DEFINE_float('keep_prob', 0.8, 'Keep probability for convnet')
flags.DEFINE_bool('render', True, 'Whether or not to render episodes')
flags.DEFINE_integer('hidden_size', 512, 'Size of hidden network')
flags.DEFINE_string('game', 'Pong-v0', 'Which game to play')
flags.DEFINE_integer('channels', 1, 'Number of image channels per frame after preprocessing')
flags.DEFINE_integer('image_size_x', 80, 'Width of game screen after preprocessing')
flags.DEFINE_integer('image_size_y', 80, 'Height of game screen after preprocessing')
flags.DEFINE_bool('logging', True, 'Whether or not to log')
flags.DEFINE_bool('verbose', True, 'Whether or not to be verbose')
flags.DEFINE_float('reg_param', 0.001, 'Regularization parameter for network')
flags.DEFINE_float('discount_factor', 0.99, 'Discount factor (gamma) for MDP')
flags.DEFINE_integer('max_memory', 5000, 'Max rollout recall size')

def main(_):
    # Initialize network
    policy_net = policy.PolicyNetwork()
    max_games = FLAGS.batch_size * FLAGS.epochs * FLAGS.epoch_size
    env = gym.make(FLAGS.game)
    observation = env.reset()
    observation = utils.preprocess_image(observation, game=FLAGS.game)

    if FLAGS.render: env.render()

    step = 0
    games_won, games_lost = 0, 0
    while step < max_games or FLAGS.epochs == -1:
        if FLAGS.render: env.render()

        state = observation
        action = policy_net.get_action(observation)
        observation, reward, done, info = env.step(action)
        observation = utils.preprocess_image(observation, game=FLAGS.game)
        policy_net.update_memory(state, action, reward)

        if reward > 0:
            games_won += 1
        else:
            games_lost += 1

        if done:
            env.reset()
            update_gradients = step % FLAGS.batch_size == 0

            print 'Finished episode %d, updating rewards and gradients...' % step
            loss, rewards = policy_net.partial_fit_step(train_batch=update_gradients)

            print 'Episode %d of %d' % (step, max_games)
            print '=' * 40
            print 'Loss = %f' % loss
            print 'Reward total = %f' % sum(rewards)
            print 'Win Percentage: %f' % (100.0 * float(games_won)/(games_won + games_lost))
            print

            games_won, games_lost = 0, 0
            step += 1


if __name__ == '__main__':
    tf.app.run()
