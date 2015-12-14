"""
this is a modified sequence 2 sequence framework for variational LSTMs.

runs this PTB dataset from Tomas Mikolov's webpage:
http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

e.g. to train the 'small' run:
python vae_lstm.py \
  --data_path=/home/altosaar/projects/arxiv/dat/simple-examples/data \
  --model small \

on either GPU or CPU.

hyperparameters used in the model:
- z_dim - latent dimensionality
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

OLD:
To compile on CPU:
  bazel build -c opt tensorflow/models/rnn/ptb:ptb_word_lm
To compile on GPU:
  bazel build -c opt tensorflow --config=cuda \
    tensorflow/models/rnn/ptb:ptb_word_lm
To run:
  ./bazel-bin/.../ptb_word_lm --data_path=/tmp/simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import sys, os
sys.path.append("/Users/jaanaltosaar/projects/installations/tensorflow")

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
#from tensorflow.models.rnn.ptb import reader
from tensorflow.models.rnn import rnn
from tensorflow.python.platform import gfile
import logging

import reader

# set seed
tf.set_random_seed(98765)

flags = tf.flags
# logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data path")
flags.DEFINE_string("checkpoint_dir", None, 'checkpoint dir to load')
flags.DEFINE_boolean('debug', False, 'debugging mode or not')
flags.DEFINE_string('out_dir', None, "output directory")

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


def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None, config=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: if not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x cell.output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x cell.input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x cell.output_size] containing generated outputs.
    states: The state of each cell in each time-step. This is a list with
      length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
      (Note that in some cases, like basic RNN cell or GRU cell, outputs and
       states can be the same. They are different for LSTM cells though.)
  """
  with tf.variable_scope(scope or "rnn_decoder"):
    states = [initial_state]
    outputs = []
    prev = None
    for i in xrange(len(decoder_inputs)):
      inp = decoder_inputs[i]
      # inp = tf.Print(inp, [inp.get_shape()])
      # prev_z_sample = tf.slice(inp, [config.batch_size, cell.input_size], [config.batch_size, -1])
      if loop_function is not None and prev is not None:
        with tf.variable_scope("loop_function", reuse=True):
          # We do not propagate gradients over the loop function.
          inp = tf.stop_gradient(loop_function(prev, i))
          # inp = tf.concat(1, [inp, tf.zeros(prev_z_sample])
      if i > 0:
        tf.get_variable_scope().reuse_variables()
      output, new_state = cell(inp, states[-1])
      outputs.append(output)
      states.append(new_state)
      if loop_function is not None:
        prev = tf.stop_gradient(output)
  return outputs, states

def vae_decoder(decoder_inputs, z_samples, initial_state, cell, loop_function=None,
                  scope=None, config=None):
  """modified RNN decoder for the VAE sequence-to-sequence model.

  Args:
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    z_samples: [batch_size x config.z_dim]
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: if not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x cell.output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x cell.input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x cell.output_size] containing generated outputs.
    states: The state of each cell in each time-step. This is a list with
      length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
      (Note that in some cases, like basic RNN cell or GRU cell, outputs and
       states can be the same. They are different for LSTM cells though.)
  """
  with tf.variable_scope(scope or "rnn_decoder"):
    states = [initial_state]
    outputs = []
    prev = None
    for i in xrange(len(decoder_inputs)):
      inp = decoder_inputs[i]
      inp_z_sample = z_samples[i]
      inp = tf.concat(1, [inp, inp_z_sample])
      if loop_function is not None and prev is not None:
        with tf.variable_scope("loop_function", reuse=True):
          # We do not propagate gradients over the loop function.
          inp = tf.stop_gradient(loop_function(prev, inp_z_sample))
      if i > 0:
        tf.get_variable_scope().reuse_variables()
      output, new_state = cell(inp, states[-1])
      outputs.append(output)
      states.append(new_state)
      if loop_function is not None:
        prev = tf.stop_gradient(output)
  return outputs, states

def rnn_decoder_argmax(decoder_inputs, initial_state, cell, num_symbols, output_projection=None, feed_previous=False, scope=None, config=None):
  if output_projection is not None:
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=tf.float32)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size,
                                                        num_symbols])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=tf.float32)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with tf.variable_scope(scope or "rnn_decoder_argmax"):

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [num_symbols, cell.input_size])

    def extract_argmax_and_embed(prev, i):
      """Loop_function that extracts the symbol from prev and embeds it."""
      print(i)
      if output_projection is not None:
        # prev_z_sample = tf.slice(prev, [config.batch_size, cell.input_size], [config.batch_size, config.z_dim])
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      prev_embedding = tf.nn.embedding_lookup(embedding, prev_symbol)
      # prev_embedding_and_z_sample = tf.concat(1, [prev_embedding, tf.zeros([1, config.z_dim])])
      # prev_embedding_and_z_sample = tf.concat(1, [prev_embedding, prev_z_sample])
      return prev_embedding

    loop_function = None
    if feed_previous:
      loop_function = extract_argmax_and_embed

    return rnn_decoder(decoder_inputs, initial_state, cell,
                       loop_function=loop_function, scope=scope, config=config)

def vae_decoder_argmax(decoder_inputs, z_samples, initial_state, cell, num_symbols, output_projection=None, feed_previous=False, scope=None, config=None):
  ### DECODER FOR VAE WITH SEPARATE Z_SAMPLES
  if output_projection is not None:
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=tf.float32)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size,
                                                        num_symbols])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=tf.float32)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with tf.variable_scope(scope or "vae_decoder_argmax"):
    with tf.variable_scope("embedding"):
      with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [num_symbols, cell.input_size])

    def extract_argmax_and_embed(prev, prev_z_sample):
      """Loop_function that extracts the symbol from prev and embeds it."""
      if output_projection is not None:
        # prev_z_sample = tf.slice(prev, [config.batch_size, cell.input_size], [config.batch_size, config.z_dim])
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      prev_embedding = tf.nn.embedding_lookup(embedding, prev_symbol)
      # prev_embedding_and_z_sample = tf.concat(1, [prev_embedding, tf.zeros([1, config.z_dim])])
      prev_embedding_and_z_sample = tf.concat(1, [prev_embedding, prev_z_sample])
      return prev_embedding_and_z_sample

    loop_function = None
    if feed_previous:
      loop_function = extract_argmax_and_embed

    return vae_decoder(decoder_inputs, z_samples, initial_state, cell,
                       loop_function=loop_function, scope=scope, config=config)

def embedding_rnn_decoder(decoder_inputs, initial_state, cell, num_symbols,
                          output_projection=None, feed_previous=False,
                          scope=None):
  """RNN decoder with embedding and a pure-decoding option.

  Args:
    decoder_inputs: a list of 1D batch-sized int32-Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: integer, how many symbols come into the embedding.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [cell.output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      If False, decoder_inputs are used as given (the standard decoder case).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x cell.output_size] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when output_projection has the wrong shape.
  """
  if output_projection is not None:
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=tf.float32)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size,
                                                        num_symbols])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=tf.float32)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with tf.variable_scope(scope or "embedding_rnn_decoder"):
    with tf.device("/cpu:0"):
      with tf.variable_scope("embedding"):
        embedding = tf.get_variable("embedding", [num_symbols, cell.input_size], reuse=True)

    def extract_argmax_and_embed(prev, _):
      """Loop_function that extracts the symbol from prev and embeds it."""
      if output_projection is not None:
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      return tf.nn.embedding_lookup(embedding, prev_symbol)

    loop_function = None
    if feed_previous:
      loop_function = extract_argmax_and_embed

    emb_inp = [tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]
    return rnn_decoder(emb_inp, initial_state, cell,
                       loop_function=loop_function)

# printing shape of a tensor live!
def pprint(tensor, message='', shape=False):
  if shape:
    tensor_shape = tensor.get_shape()
    to_print = [dim for dim in tensor_shape if dim != None]
    print(to_print)
    return tf.Print(tensor, [tensor.get_shape(), to_print], message)
  else:
    return tf.Print(tensor, [tensor], message)

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, decode_only=False):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    with tf.variable_scope("cell_encoder"):
      lstm_encoder_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
      if is_training and config.keep_prob < 1:
        lstm_encoder_cell = rnn_cell.DropoutWrapper(
            lstm_encoder_cell, output_keep_prob=config.keep_prob)
      cell_encoder = rnn_cell.MultiRNNCell([lstm_encoder_cell] * config.num_layers)

      # this is the linear projection layer down to num_encoder_symbols = 2*config.z_dim
      cell_encoder = rnn_cell.OutputProjectionWrapper(cell_encoder, 2 * config.z_dim)

      self._initial_state_encoder = cell_encoder.zero_state(batch_size, tf.float32)


    with tf.variable_scope("cell_decoder"):
      lstm_decoder_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
      if is_training and config.keep_prob < 1:
        lstm_decoder_cell = rnn_cell.DropoutWrapper(
            lstm_decoder_cell, output_keep_prob=config.keep_prob)
      cell_decoder = rnn_cell.MultiRNNCell([lstm_decoder_cell] * config.num_layers)

      self._initial_state_decoder = cell_decoder.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      with tf.variable_scope("embedding"):
        embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.split(
          1, num_steps, tf.nn.embedding_lookup(embedding, self._input_data))
      inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    if is_training and config.keep_prob < 1:
      inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

    #inputs[0] = tf.Print(inputs[0], [inputs[0].get_shape(), inputs[0]])
    # inputs[0] = pprint(inputs[0], shape=True, message='inputs')

    # initial inputs
    inputs_encoder = inputs
    # inputs_encoder[0] = pprint(inputs_encoder[0], message='inputs to encoder')

    outputs_encoder, states_encoder = rnn.rnn(cell_encoder, inputs_encoder, initial_state=self._initial_state_encoder)

    # outputs_encoder[0] = pprint(outputs_encoder[0], message='encoder outputs')

    # split the outputs to mu and log_sigma
    mu_and_log_sigmas = [tf.split(1, 2, output_encoder) for output_encoder in outputs_encoder]
    mus = [mu_and_log_sigma[0] for mu_and_log_sigma in mu_and_log_sigmas]
    # print(mus[0].get_shape())
    # mus[0] = pprint(mus[0], message='mu')
    log_sigmas = [mu_and_log_sigma[1] for mu_and_log_sigma in mu_and_log_sigmas]

    # epsilon is sampled from N(0,1) for location-scale transform
    epsilons = [tf.random_normal([config.batch_size, config.z_dim]) for i in range(len(log_sigmas))]

    # do the location-scale transform
    z_samples = [tf.add(mu, tf.mul(tf.exp(log_sigma), epsilon)) for mu, log_sigma, epsilon in zip(mus, log_sigmas, epsilons)]
    if decode_only:
      # if we're decoding, just sample from a random normal
      z_samples = [tf.random_normal([1, config.z_dim]) for i in range(len(z_samples))]
    # z_samples[0] = pprint(z_samples[0], message='z_sample')

    # calculate KL. equation 10 from kingma - auto-encoding variational bayes.
    neg_KL_list = [tf.add_n([tf.ones_like(mu), tf.log(tf.square(tf.exp(log_sigma))), tf.neg(tf.square(mu)), tf.neg(tf.square(tf.exp(log_sigma)))]) for mu, log_sigma in zip(mus, log_sigmas)]

    # multiply by 0.5
    neg_KL_list = [tf.mul(tf.constant(0.5, shape=[1, config.z_dim]), KL_term) for KL_term in neg_KL_list]

    # merge the list like we merge the outputs
    neg_KL = tf.reshape(tf.concat(1, neg_KL_list), [-1, config.z_dim])

    # neg_KL = pprint(neg_KL, message='neg_KL')

    # print(decoder_inputs[0:10])#, [decoder_inputs[0]])

    # decoder_inputs[0] = pprint(decoder_inputs[0], message='decoder_input')
    # decoder_inputs[0] = tf.Print(decoder_inputs[0], [decoder_inputs[0].get_shape()])

    # no pure decoding opt
    # outputs_decoder, states_decoder = rnn_decoder(decoder_inputs, self._initial_state_decoder, cell_decoder)

    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])

    # concatenate z_samples with previous timesteps
    # decoder_inputs = [tf.concat(1, [single_input, z_sample]) for single_input, z_sample in zip(inputs_encoder, z_samples)]
    # outputs_decoder, states_decoder = rnn_decoder_argmax(decoder_inputs, self._initial_state_decoder, cell_decoder, vocab_size,
    #   output_projection=[softmax_w, softmax_b],
    #   feed_previous=True,
    #   config=config)

    # refactored to be like sam's
    # z_samples = [tf.reshape(z_sample, [1, config.z_dim]) for z_sample in z_samples]
    outputs_decoder, states_decoder = vae_decoder_argmax(inputs_encoder, z_samples, self._initial_state_decoder, cell_decoder, vocab_size,
      output_projection=[softmax_w, softmax_b],
      feed_previous=True,
      config=config)


    # outputs_decoder[0] = pprint(outputs_decoder[0], message='outputs from decoder')
    # final output
    outputs = outputs_decoder
    # outputs[0] = pprint(outputs[0], message='outputs')

    # print(len(outputs))
    # print(outputs[0].get_shape())
    # outputs = []
    # states = []
    # state = self._initial_state
    # with tf.variable_scope("RNN"):
    #   for time_step, input_ in enumerate(inputs):
    #     if time_step > 0: tf.get_variable_scope().reuse_variables()
    #     (cell_output, state) = cell(input_, state)
    #     outputs.append(cell_output)
    #     states.append(state)

    # do a softmax over the vocabulary using the decoder outputs!
    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    logits = tf.nn.xw_plus_b(output,
                             softmax_w,
                             softmax_b)
    loss = seq2seq.sequence_loss_by_example([logits],
                                            [tf.reshape(self._targets, [-1])],
                                            [tf.ones([batch_size * num_steps])],
                                            vocab_size)
    loss_scalar = tf.reduce_sum(loss)
    KL_scalar = tf.neg(tf.reduce_sum(neg_KL))
    # here we compute the *NEGATIVE* ELBO (because we don't know how the optimizer deals with negative learning rates / gradients)
    # the loss in seq2seq.sequence_loss_by_example is the cross-entropy, which is the *negative* log-likelihood, so we can add it.
    cost = KL_scalar + loss_scalar# / batch_size
    # cost = pprint(cost, message='ELBO')
    self._cost = cost
    self._final_state = states_encoder[-1]
    if decode_only:
      self._logits = logits
      return

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    # optimizer = tf.train.AdamOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state_encoder

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def logits(self):
    return self._logits

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5 #grad clippin
  num_layers = 1
  num_steps = 20#20
  hidden_size = 200#100# 2 for debugging
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  z_dim = 50# 1 for debugging

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 1 #2
  num_steps = 35
  hidden_size = 650 #650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000



class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 1#20
  vocab_size = 10000


def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    # print(x)
    # print('^inputs')
    # print(y)
    # print('^targets')
    costs += cost
    iters += m.num_steps
    if verbose and step % (epoch_size // 10) == 10:
    # if verbose and step % 10 == 0:
      logging.info("%.3f ELBO: %.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, costs / iters, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def train(unused_args):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  if FLAGS.debug:
    train_data = train_data[0:2342]
    valid_data = valid_data[0:1332]
    test_data = test_data[0:987]
    logging.info('train data length is ', len(train_data))

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config)
      mtest = PTBModel(is_training=False, config=eval_config)

    # create saver to checkpoint
    saver = tf.train.Saver()
    if FLAGS.checkpoint_dir:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        # Restores from checkpoint
        saver.restore(session, ckpt.model_checkpoint_path)
        print('loaded checkpoint from %s' % ckpt.model_checkpoint_path)
    else:
      tf.initialize_all_variables().run()
      logging.info('not using checkpoint file found; initialized vars')

    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      logging.info("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      # if not FLAGS.checkpoint_dir:
      train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                     verbose=True)
      # else:
        # train_perplexity = 0
      save_path = saver.save(session, "{}model.ckpt".format(FLAGS.out_dir))
      logging.info("Model saved in file: %s" % save_path)
      logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      logging.info("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    logging.info("Test Perplexity: %.3f" % test_perplexity)

def decode():
  # given a trained model's checkpoint file, this function generates sample sentences

  # from tensorflow.models.rnn.translate.translate.py
  # get the vocab
  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, word_to_id = raw_data
  id_to_word = {v:k for k, v in word_to_id.items()}
  vocabulary = len(word_to_id)

  config = get_config()
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      config.batch_size = 1
      m = PTBModel(is_training=False, config=config, decode_only=True)

    # create saver to checkpoint
    saver = tf.train.Saver()
    if FLAGS.checkpoint_dir:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        # Restores from checkpoint
        saver.restore(session, ckpt.model_checkpoint_path)
        logging.info('loaded checkpoint from %s' % ckpt.model_checkpoint_path)
    else:
      logging.info('no checkpoint file found; need this to decode')
      return

    # Decode from standard input.
    sentence_length = config.num_steps

    state = m.initial_state.eval()
    for sentence_idx in range(20):
      print('sample {}'.format(sentence_idx))
      # only the first word is used during decoding; the rest are ignored using the loop_function in vae_decoder
      x = np.floor(np.random.rand(1,sentence_length)*config.vocab_size).astype(np.int32)
      # cost, state, _ = session.run([m.cost, m.final_state, tf.no_op()],
      #                              {m.input_data: x,
      #                               m.targets: x,
      #                               m.initial_state: state})
      # print(cost)
      logits = session.run([m.logits],{m.input_data: x,
                                        m.targets: x,
                                        m.initial_state: state})

      word_ids = [int(np.argmax(logit, axis=0)) for logit in logits[0]]
      sentence = [id_to_word[word_id] for word_id in word_ids]
      sentence_str = ' '.join(sentence)
      print(sentence_str)

def main(unused_args):
  if FLAGS.checkpoint_dir:
    logging.info('decoding from: ')
    logging.info(FLAGS.checkpoint_dir)
    decode()
  else:
    train(unused_args)

if __name__ == "__main__":
    tf.app.run()
