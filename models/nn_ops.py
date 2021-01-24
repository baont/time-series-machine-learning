#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import tensorflow as tf
from tensorflow.python.training import moving_averages


def leaky_relu(x, alpha=0.1):
  x = tf.compat.v1.nn.relu(x)
  m_x = tf.compat.v1.nn.relu(-x)
  x -= alpha * m_x
  return x


def prelu(x):
  shape = x.get_shape()
  alpha = tf.compat.v1.Variable(initial_value=tf.compat.v1.zeros(shape=shape[1:]), name='alpha')
  x = tf.compat.v1.nn.relu(x) + tf.compat.v1.multiply(alpha, (x - tf.compat.v1.abs(x))) * 0.5
  return x


ACTIVATIONS = {'leaky_relu': leaky_relu, 'prelu': prelu}
ACTIVATIONS.update({name: getattr(tf, name) for name in ['tanh']})
ACTIVATIONS.update({name: getattr(tf.compat.v1.nn, name) for name in ['relu', 'elu', 'sigmoid']})


COST_FUNCTIONS = {
  'l1': lambda output, y: tf.compat.v1.reduce_mean(tf.compat.v1.abs(output - y)),
  'l2': lambda output, y: tf.compat.v1.reduce_mean(tf.compat.v1.pow(output - y, 2.0)),
}


def dropout(incoming, is_training, keep_prob):
  if keep_prob is None:
    return incoming
  return tf.compat.v1.cond(is_training, lambda: tf.compat.v1.nn.dropout(incoming, keep_prob), lambda: incoming)


def batch_normalization(incoming, is_training, beta=0.0, gamma=1.0, epsilon=1e-5, decay=0.9):
  shape = incoming.get_shape()
  dimensions_num = len(shape)
  axis = list(range(dimensions_num - 1))

  with tf.compat.v1.compat.v1.variable_scope('batchnorm'):
    beta = tf.compat.v1.Variable(initial_value=tf.compat.v1.ones(shape=[shape[-1]]) * beta, name='beta')
    gamma = tf.compat.v1.Variable(initial_value=tf.compat.v1.ones(shape=[shape[-1]]) * gamma, name='gamma')

    moving_mean = tf.compat.v1.Variable(initial_value=tf.compat.v1.zeros(shape=shape[-1:]), trainable=False, name='moving_mean')
    moving_variance = tf.compat.v1.Variable(initial_value=tf.compat.v1.zeros(shape=shape[-1:]), trainable=False, name='moving_variance')

  def update_mean_var():
    mean, variance = tf.compat.v1.nn.moments(incoming, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
    with tf.compat.v1.control_dependencies([update_moving_mean, update_moving_variance]):
      return tf.compat.v1.identity(mean), tf.compat.v1.identity(variance)

  mean, var = tf.compat.v1.cond(is_training, update_mean_var, lambda: (moving_mean, moving_variance))
  inference = tf.compat.v1.nn.batch_normalization(incoming, mean, var, beta, gamma, epsilon)
  inference.set_shape(shape)
  return inference
