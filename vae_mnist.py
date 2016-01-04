''' port https://jmetzen.github.io/2015-11-27/vae.html for GPUs on tiger '''

import sys, os
sys.path.append("/Users/jaanaltosaar/projects/installations/tensorflow")

import tensorflow.python.platform

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data #LOCAL
import logging, time
from collections import OrderedDict

flags = tf.flags

flags.DEFINE_string('out_dir', None, "output directory")
flags.DEFINE_boolean('mirror_autoencoder', False, 'whether to use mirror autoencoder training')

FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.out_dir):
  os.makedirs(FLAGS.out_dir)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(FLAGS.out_dir + 'job.log', 'w')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

from tensorflow.python.platform import logging

logging.info('FLAGS.out_dir: {}'.format(FLAGS.out_dir))

np.random.seed(0)
tf.set_random_seed(0)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_samples = mnist.train.num_examples

# printing shape of a tensor live!
def pprint(tensor, message='', shape=False):
  if shape:
    tensor_shape = tensor.get_shape()
    to_print = [dim for dim in tensor_shape if dim != None]
    print(to_print)
    return tf.Print(tensor, [tensor.get_shape(), to_print], message)
  else:
    return tf.Print(tensor, [tensor], message)

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100, wq_gradient_steps=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.wq_gradient_steps = wq_gradient_steps

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create autoencoder network
        self.network_weights = self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

        # initialize summary writer
        self.writer = tf.train.SummaryWriter(FLAGS.out_dir, self.sess.graph_def)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

        return network_weights

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1,  n_hidden_gener_2,
                            n_input, n_z):
        all_weights = OrderedDict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([1, n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([1, n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([1, n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([1, n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        def calc_ELBO(weights_recog, biases_recog):
            # calc_ELBO needs to depend on wq (i.e. weights_recog, biases_recog)
            # Use recognition network to determine mean and
            # (log) variance of Gaussian distribution in latent
            # space
            self.z_mean, self.z_log_sigma_sq = \
                self._recognition_network(weights_recog, biases_recog)

            # Draw one sample z from Gaussian distribution
            n_z = self.network_architecture["n_z"]
            eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)
            # z = mu + sigma*epsilon
            self.z = tf.add(self.z_mean,
                            tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

            # The loss is composed of two terms:
            # 1.) The reconstruction loss (the negative log probability
            #     of the input under the reconstructed Bernoulli distribution
            #     induced by the decoder in the data space).
            #     This can be interpreted as the number of "nats" required
            #     for reconstructing the input when the activation in latent
            #     is given.
            # Adding 1e-10 to avoid evaluatio of log(0.0)
            # Use generator to determine mean of
            # Bernoulli distribution of reconstructed input

            reconstr_loss = \
                -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                               + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                               1)
            # 2.) The latent loss, which is defined as the Kullback Leibler divergence
            ##    between the distribution in latent space induced by the encoder on
            #     the data and some prior. This acts as a kind of regularizer.
            #     This can be interpreted as the number of "nats" required
            #     for transmitting the the latent space distribution given
            #     the prior.
            latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                               - tf.square(self.z_mean)
                                               - tf.exp(self.z_log_sigma_sq), 1)

            ELBO = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
            return ELBO, latent_loss, reconstr_loss

        ELBO, KL, NLL = calc_ELBO(self.network_weights['weights_recog'], self.network_weights['biases_recog'])

        # Compute the gradients for a list of variables.
        # tvars = tf.trainable_variables()

        # isolate recognition / encoder network weights wq
        tvars_wq_weights = self.network_weights['weights_recog']
        tvars_wq_biases = self.network_weights['biases_recog']
        tvars_wp = self.network_weights['weights_gener'].values() + self.network_weights['biases_gener'].values()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)#.minimize(self.cost)

        if FLAGS.mirror_autoencoder:
            # get gradients w.r.t. wp, save these for later
            grads_ELBO_wp_unclipped = tf.gradients(ELBO, tvars_wp)

            # initialize wt to be the old wq + some noise.
            wt_weights = {k : var + tf.random_normal(var.get_shape(), 0, 0.01, dtype=tf.float32) for k, var in tvars_wq_weights.items()}
            wt_biases = {k : var + tf.random_normal(var.get_shape(), 0, 0.01, dtype=tf.float32) for k, var in tvars_wq_biases.items()}

            # these weights are what we update using the update equation U
            w_weights = tvars_wq_weights
            w_biases = tvars_wq_biases

            # initialize old wt vars
            # first term of U(wq) is ELBO evaluated at old weights wt.
            ELBO_wt, KL_wt, _ = calc_ELBO(wt_weights, wt_biases)

            # w are the variables we will now train.
            w = w_weights.values() + w_biases.values()

            # update wq weights for some steps
            for wq_step in range(0, self.wq_gradient_steps):
                print('wq_step ', wq_step)
                wt = wt_weights.values() + wt_biases.values()

                # compute w - wt
                w_minus_wt = [w_var - wt_var for w_var, wt_var in zip(w, wt)]

                # first order expansion term needs the gradient of the ELBO wrt both weights and biases:
                # important Q: is the gradient wrt w_q of the ELBO evaluated at wt a scalar? the ELBO is a scalar, but w is a vector, hence grad(ELBO)|_wt should be a vector. hence we need its global norm in the taylor expansion?
                grads_ELBO_wt = tf.gradients(ELBO_wt, w)

                # reshape grads, they have Dimension(None) for some reason!
                for i, grad in enumerate(grads_ELBO_wt):
                    grads_ELBO_wt[i].set_shape(w_minus_wt[i].get_shape())

                # first order term is grads dot w_minus_wt. we need to flatten them to take the proper dot product.
                grads_ELBO_wt_flat = [tf.reshape(g, [1, -1]) for g, v in zip(grads_ELBO_wt, w_minus_wt)]
                w_minus_wt_flat = [tf.reshape(v, [1, -1]) for v in w_minus_wt]
                dot_product_list = [tf.batch_matmul(g, v, adj_y=True) for \
                                g, v in zip(grads_ELBO_wt_flat, w_minus_wt_flat)]
                first_order_term = tf.add_n(dot_product_list)

                # mean_squared term needs c/2, lr_2, w_minus_wt_sq
                c_div_2 = self.learning_rate / 2
                w_minus_wt_sq = [tf.reduce_sum(tf.square(w_minus_wt_var)) for w_minus_wt_var in w_minus_wt]
                mean_squared_term = c_div_2 * tf.add_n(w_minus_wt_sq)

                # KL_diff_term needs k/2, lr_2
                k_div_2 = self.learning_rate / 2
                # we also need to calculate the ELBO_w because it'll be the next step's ELBO_wt!
                ELBO_w, KL_w, _ = calc_ELBO(w_weights, w_biases)
                KL_diff_term = k_div_2 * tf.square(KL_w - KL_wt)

                # add all terms to get update equation U. tested grads for KL_diff_term.
                U_w = first_order_term + mean_squared_term + KL_diff_term
                # U_w = first_order_term

                # take gradient w.r.t. w, wq weights.
                grads_U_w_unclipped = tf.gradients(U_w, w)

                # update wq weights
                self.optimizer.apply_gradients(zip(grads_U_w_unclipped, w))

                # set wt = w for next iteration.
                wt_weights = w_weights.copy()
                wt_biases = w_biases.copy()
                ELBO_wt = ELBO_w
                KL_wt = KL_w

            # apply old gradients for wp:
            self.train_op = self.optimizer.apply_gradients(zip(grads_ELBO_wp_unclipped, tvars_wp))

            # when all is said and done, define the cost according to the final ELBO!
            self.cost = ELBO
        else:
            # append generative / decoder network weights wp
            print 'appending generative weights wp'
            tvars = tvars_wq_weights.values() + tvars_wq_biases.values() + tvars_wp
            grads_unclipped = tf.gradients(ELBO, tvars)
            # Use ADAM optimizer
            self.train_op = self.optimizer.apply_gradients(zip(grads_unclipped, tvars))
            self.cost = ELBO

        self.KL = tf.reduce_mean(KL)
        self.NLL = tf.reduce_mean(NLL)

        '''calculate number of approximate active units as in burda.
        i.e. take the expectation of the draw from the posterior (the mean in this gaussian case).
        then calculate the variance, and threshold it heuristically.
        ref: https://github.com/yburda/iwae/blob/master/iwae.py#L268 and L288
        '''
        self.z_mean.set_shape([self.batch_size, self.z_mean.get_shape()[1]])
        _, variance_of_batch_means = tf.nn.moments(self.z_mean, axes=[0])
        thresholded = tf.greater(variance_of_batch_means, tf.constant(0.01,
                                 dtype=tf.float32, shape=variance_of_batch_means.get_shape()))
        thresholded_float = tf.cast(thresholded, dtype=tf.float32)
        num_active = tf.reduce_sum(thresholded_float)
        num_active = tf.Print(num_active, [num_active], 'active latent dimensions: ')

        active_unit_summary = tf.scalar_summary('approximate number of active units', num_active)
        KL_summary = tf.scalar_summary('KL', self.KL)
        NLL_summary = tf.scalar_summary('NLL', self.NLL)
        neg_ELBO_summary = tf.scalar_summary("neg_ELBO", self.cost)

        merged = tf.merge_all_summaries()
        self.merged = merged

    def partial_fit(self, X, write_summary=False, global_step=None):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        if write_summary:
          merged, train_op, cost, KL, NLL = self.sess.run((self.merged, self.train_op, self.cost, self.KL, self.NLL), feed_dict={self.x: X})
          logging.info('adding summary, global step {}'.format(global_step))
          self.writer.add_summary(merged, global_step=global_step)
        else:
          train_op, cost, KL, NLL = self.sess.run((self.train_op, self.cost, self.KL, self.NLL), feed_dict={self.x: X})

        cost_dict = {'ELBO': cost,
                      'KL': KL,
                      'NLL': NLL}
        return cost_dict

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})

def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=50, display_step=1):

    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size, transfer_fct=tf.nn.relu)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost_dict = {'ELBO': 0., 'KL': 0., 'NLL': 0.}
        total_batch = int(n_samples / batch_size)
        start_time = time.time()

        # Loop over all batches
        for i in range(total_batch):

            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            # write summary every 10k samples
            global_step = i + total_batch * epoch
            if global_step % 500 == 0:
              cost_dict = vae.partial_fit(batch_xs, write_summary=True, global_step=global_step)
            else:
              cost_dict = vae.partial_fit(batch_xs)
            for cost_name in avg_cost_dict.keys():
              # Compute average loss in the dict
              avg_cost_dict[cost_name] += cost_dict[cost_name] / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            logging.info("Epoch: {0}, ELBO={1:.3f}, KL={2:.3f}, NLL={3:.3f}, speed: {4:.0f} samples per second".format(
                         epoch+1, avg_cost_dict['ELBO'],
                         avg_cost_dict['KL'], avg_cost_dict['NLL'],
                         total_batch * batch_size / (time.time() - start_time)))
    return vae

network_architecture = \
    dict(n_hidden_recog_1=50, # 1st layer encoder neurons
         n_hidden_recog_2=50, # 2nd layer encoder neurons
         n_hidden_gener_1=50, # 1st layer decoder neurons
         n_hidden_gener_2=50, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=50)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=250, batch_size=100)