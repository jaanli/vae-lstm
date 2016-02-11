"""
Modified version of LSTM language modeling for sequence embedding

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

import reader
import lstm


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
        "model", "small",
        "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("out_dir", None, 'out_dir"')
flags.DEFINE_string("checkpoint_file", None, 'checkpoint file to load')

FLAGS = flags.FLAGS

run_dir = os.path.join(FLAGS.out_dir, 'model_{model}'.format(
    model=FLAGS.model))

if not os.path.exists(run_dir):
    os.makedirs(run_dir)


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.1 #1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 20
    keep_prob = 1.0
    lr_decay = 1.0
    batch_size = 20
    vocab_size = 72094


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 72094


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
    batch_size = 20
    vocab_size = 72094


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
        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
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


def main(unused_args):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = lstm.LSTMSeqEmbedModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = lstm.LSTMSeqEmbedModel(is_training=False, config=config)
            mtest = lstm.LSTMSeqEmbedModel(is_training=False, config=eval_config)

        saver = tf.train.Saver()

        if FLAGS.checkpoint_file:
            saver.restore(session, FLAGS.checkpoint_file)
            print("Loaded checkpoint from %s" % FLAGS.checkpoint_file)
            #embedding = m.embedding.eval()
            #np.save("embedding_LSTM.npy", embedding)
        else:
            tf.initialize_all_variables().run()

            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1,
                                                         session.run(m.lr)))
                train_perplexity = run_epoch(session, m, train_data,
                                             m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1,
                                                            train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, valid_data,
                                             tf.no_op())
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1,
                                                            valid_perplexity))

                save_path = saver.save(session, os.path.join(run_dir,
                                                             "model.ckpt"))
                print("Model saved in file: %s" % save_path)

        test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()
