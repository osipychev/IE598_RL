# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import numpy as np
import tensorflow as tf

from Config import Config


class NetworkVP:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                

    def _create_graph(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')
        self.advantages = tf.placeholder(tf.float32, [None], name='advantages')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        # Yizhi edit here
        # As implemented in A3C paper
        self.n1_a1 = self.conv2d_layer(self.x, 8, 16, 'conv11-agent1', strides=[1, 4, 4, 1])
        self.n2_a1 = self.conv2d_layer(self.n1_a1, 4, 32, 'conv12-agent1', strides=[1, 2, 2, 1])
        self.action_index1 = tf.placeholder(tf.float32, [None, self.num_actions])
        _input1 = self.n2_a1

        self.n1_a2 = self.conv2d_layer(self.x, 8, 16, 'conv11-agent2', strides=[1, 4, 4, 1])
        self.n2_a2 = self.conv2d_layer(self.n1_a2, 4, 32, 'conv12-agent2', strides=[1, 2, 2, 1])
        self.action_index2 = tf.placeholder(tf.float32, [None, self.num_actions])
        _input2 = self.n2_a2

        flatten_input_shape1 = _input1.get_shape()
        nb_elements1 = flatten_input_shape1[1] * flatten_input_shape1[2] * flatten_input_shape1[3]

        flatten_input_shape2 = _input2.get_shape()
        nb_elements2 = flatten_input_shape2[1] * flatten_input_shape2[2] * flatten_input_shape2[3]

        self.flat1 = tf.reshape(_input1, shape=[-1, nb_elements1._value])
        self.d1_a1 = self.dense_layer(self.flat1, 256, 'dense1-agent1')

        self.flat2 = tf.reshape(_input2, shape=[-1, nb_elements2._value])
        self.d1_a2 = self.dense_layer(self.flat2, 256, 'dense1-agent2')

        self.d1 = tf.concat([self.d1_a1, self.d1_a2], axis=1)

        self.logits_v = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v', func=None), axis=[1])
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

        self.logits_p1 = self.dense_layer(self.d1_a1, self.num_actions, 'logits_p-agent1', func=None)
        self.logits_p2 = self.dense_layer(self.d1_a2, self.num_actions, 'logits_p-agent2', func=None)

        self.softmax_p1 = (tf.nn.softmax(self.logits_p1) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
        self.softmax_p2 = (tf.nn.softmax(self.logits_p2) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
        self.selected_action_prob1 = tf.reduce_sum(self.softmax_p1 * self.action_index1, axis=1)
        self.selected_action_prob2 = tf.reduce_sum(self.softmax_p2 * self.action_index2, axis=1)

        self.cost_p1 = tf.log(tf.maximum(self.selected_action_prob1, self.log_epsilon)) \
                            * self.advantages
        self.cost_p1 += -1 * self.var_beta * \
                            tf.reduce_sum(tf.log(tf.maximum(self.softmax_p1, self.log_epsilon)) * self.softmax_p1, axis=1)

        self.cost_p2 = tf.log(tf.maximum(self.selected_action_prob2, self.log_epsilon)) \
                            * self.advantages
        self.cost_p2 += -1 * self.var_beta * \
                            tf.reduce_sum(tf.log(tf.maximum(self.softmax_p2, self.log_epsilon)) * self.softmax_p2, axis=1)
        
        self.cost_p1 = -tf.reduce_sum(self.cost_p1, axis=0)
        self.cost_p2 = -tf.reduce_sum(self.cost_p2, axis=0)
        
        self.opt = tf.train.RMSPropOptimizer(
                        learning_rate=self.var_learning_rate,
                        decay=Config.RMSPROP_DECAY,
                        momentum=Config.RMSPROP_MOMENTUM,
                        epsilon=Config.RMSPROP_EPSILON)

        self.train_op_p1 = self.opt.minimize(self.cost_p1, global_step=self.global_step)
        self.train_op_p2 = self.opt.minimize(self.cost_p2, global_step=self.global_step)
        self.train_op_v = self.opt.minimize(self.cost_v, global_step=self.global_step)
        self.train_op = [self.train_op_p1, self.train_op_p2, self.train_op_v]

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_n1", self.n1))
        summaries.append(tf.summary.histogram("activation_n2", self.n2))
        summaries.append(tf.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])

    def predict_v(self, x):
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: x})
        return prediction

    def predict_p(self, x):
        prediction1 = self.sess.run(self.softmax_p1, feed_dict={self.x: x})
        prediction2 = self.sess.run(self.softmax_p2, feed_dict={self.x: x})
        return prediction1, prediction2
    
    def predict_p_and_v(self, x):
        return self.sess.run([self.softmax_p1, self.softmax_p2, self.logits_v], feed_dict={self.x: x})
    
    def train(self, x, y_r, adv, a1, a2, trainer_id):
        # Yizhi edit here
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.advantages: adv, self.action_index1: a1, self.action_index2: a2})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, x, y_r, adv, a1, a2):
        # Yizhi edit here
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.advantages: adv, self.action_index1: a1, self.action_index2: a2})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
