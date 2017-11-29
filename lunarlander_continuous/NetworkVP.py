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
    def __init__(self, device, model_name, action_shape, obs_shape):
        self.device = device
        self.model_name = model_name
        self.action_shape = action_shape
        self.obs_shape = obs_shape

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
        # Yizhi edit here
        self.obs1 = tf.placeholder(
            tf.float32, [None] + list(self.obs_shape), name='observation-agent1')
        self.obs2 = tf.placeholder(
            tf.float32, [None] + list(self.obs_shape), name='observation-agent2')
        self.advantages = tf.placeholder(tf.float32, [None], name='advantages')
        self.v_targets = tf.placeholder(tf.float32, [None], name='value_targets')
        self.actions1 = tf.placeholder(tf.float32, [None] + list(self.action_shape), name='actions-agent1')
        self.actions2 = tf.placeholder(tf.float32, [None] + list(self.action_shape), name='actions-agent2')
        self.old_means1 = tf.placeholder(tf.float32, [None] + list(self.action_shape), name='old_action_means-agent1')
        self.old_means2 = tf.placeholder(tf.float32, [None] + list(self.action_shape), name='old_action_means-agent2')
        self.old_log_stds1 = tf.placeholder(tf.float32, [None] + list(self.action_shape), name='old_action_log_stds-agent1')
        self.old_log_stds2 = tf.placeholder(tf.float32, [None] + list(self.action_shape), name='old_action_log_stds-agent2')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        self.global_step = tf.Variable(0, trainable=False, name='step')
      
        # std params - agent 1
        self.log_std_var1 = tf.get_variable('log_std_var-agent1', shape=(self.action_shape[0],), 
            dtype=tf.float32, initializer=tf.constant_initializer(np.log(1.0)))
        self.log_stds1 = self.param_layer(self.log_std_var1, self.obs1)
        self.log_stds1 = tf.maximum(self.log_stds1, np.log(1e-6))   # min-std to prevent numerical issues
        
        # std params - agent 2
        self.log_std_var2 = tf.get_variable('log_std_var-agent2', shape=(self.action_shape[0],), 
            dtype=tf.float32, initializer=tf.constant_initializer(np.log(1.0)))
        self.log_stds2 = self.param_layer(self.log_std_var2, self.obs2)
        self.log_stds2 = tf.maximum(self.log_stds2, np.log(1e-6))   # min-std to prevent numerical issues

        # mean network - agent 1
        self.n1_a1 = self.dense_layer(self.obs1, 64, 'dens1-agent1', func=tf.nn.tanh)
        self.n2_a1 = self.dense_layer(self.n1_a1, 64, 'dens2-agent1', func=tf.nn.tanh)
        self.means1 = self.dense_layer(self.n2_a1, 2, 'dens3-agent1', func=tf.nn.tanh)

        # mean network - agent 2
        self.n1_a2 = self.dense_layer(self.obs2, 64, 'dens1-agent2', func=tf.nn.tanh)
        self.n2_a2 = self.dense_layer(self.n1_a2, 64, 'dens2-agnet2', func=tf.nn.tanh)
        self.means2 = self.dense_layer(self.n2_a2, 2, 'dens3-agnet2', func=tf.nn.tanh)
        
        # value network for critic - agent 1 & 2 share the same critic
        self.obs = tf.concat([self.obs1, self.obs2], axis=1)
        self.cn1 = self.dense_layer(self.obs, 64, 'dens1c', func=tf.nn.tanh)
        self.cn2 = self.dense_layer(self.cn1, 64, 'dens2c', func=tf.nn.tanh)
        self.values = self.dense_layer(self.cn2, 1, 'dens3c', func=tf.nn.tanh)

        # policy gradient loss - agent 1
        self.logli1 = self.loglikelihood(self.actions1, self.means1, self.log_stds1)
        self.ent1 = self.entropy(self.log_stds1)
        self.p_loss1 = - tf.reduce_mean(self.logli1 * self.advantages, axis=0)
        self.p_loss1 -= self.var_beta * tf.reduce_mean(self.ent1)  # entropy regularization
        
        # policy gradient loss - agent 2
        self.logli2 = self.loglikelihood(self.actions2, self.means2, self.log_stds2)
        self.ent2 = self.entropy(self.log_stds2)
        self.p_loss2 = - tf.reduce_mean(self.logli2 * self.advantages, axis=0)
        self.p_loss2 -= self.var_beta * tf.reduce_mean(self.ent2)  # entropy regularization

        # value loss
        self.v_loss = tf.reduce_mean(tf.square(self.values - self.v_targets))
        
        # compute kl divergence - agent 1
        self.kl_div1 = self.kl(self.old_means1, self.old_log_stds1, self.means1, self.log_stds1)

        # compute kl divergence - agent 2
        self.kl_div2 = self.kl(self.old_means2, self.old_log_stds2, self.means2, self.log_stds2)
        
        self.opt = tf.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        self.train_op_p1 = self.opt.minimize(self.p_loss1, global_step=self.global_step)
        self.train_op_p2 = self.opt.minimize(self.p_loss2, global_step=self.global_step)
        self.train_op_v = self.opt.minimize(self.v_loss, global_step=self.global_step)
        
   
    def loglikelihood(self, actions, means, log_stds):
      # "YOUR CODE HERE"
      stds = tf.exp(log_stds)
      logli = - tf.log(2*np.pi)
      logli -= tf.reduce_sum(log_stds + 1/(2 * stds**2) * (actions - means)**2,1)
      return logli
      
    def entropy(self, log_stds):
      # "YOUR CODE HERE"
      entropy1 = tf.log(tf.sqrt(2*np.pi*np.e) * tf.exp(log_stds[0]))
      entropy2 = tf.log(tf.sqrt(2*np.pi*np.e) * tf.exp(log_stds[1]))
      return entropy1 + entropy2
      
    def kl(self, old_means, old_log_stds, new_means, new_log_stds):
      # "YOUR CODE HERE"
      new_stds = tf.exp(new_log_stds)
      old_stds = tf.exp(old_log_stds)
      kl = new_log_stds - old_log_stds + (old_stds**2 + (old_means-new_means)**2)/(2*new_stds**2) - 1/2
      return tf.reduce_sum(tf.reduce_sum(kl,1))
    
    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Policy Loss", self.p_loss))
        #summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        #summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_h1", self.h1))
        summaries.append(tf.summary.histogram("activation_h2", self.h2)) 
        summaries.append(tf.summary.histogram("advantages", self.advantages))
        summaries.append(tf.summary.histogram("log_likelihood", self.logli))
        summaries.append(tf.summary.histogram("entropy", self.ent))
        summaries.append(tf.summary.histogram("policy_means", self.means))
        summaries.append(tf.summary.histogram("policy_log_stds", self.log_stds))
        summaries.append(tf.summary.histogram("observations", self.obs))
        
        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def param_layer(self, param, input):
        ndim = input.get_shape().ndims
        reshaped_param = tf.reshape(param, (1,) * (ndim - 1) + tuple(param.get_shape().as_list()))
        tile_arg = tf.concat([tf.shape(input)[:ndim - 1], [1]], 0)
        tiled = tf.tile(reshaped_param, tile_arg)
        return tiled

    def dense_layer(self, input, out_dim, name, func=tf.nn.tanh):
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

    def predict_p_and_v(self, x1, x2):
        # Yizhi edit here
        means1, log_stds1 = self.sess.run([self.means1, self.log_stds1], feed_dict={self.obs1: x1})
        means2, log_stds2 = self.sess.run([self.means2, self.log_stds2], feed_dict={self.obs2: x2})
        values = self.sess.run(self.values, feed_dict={self.obs1: x1, self.obs2: x2})
        return means1, means2, log_stds1, log_stds2, values.reshape(-1)
    
    def train(self, x1, x2, y_r, adv, a1, a2, a_m1, a_m2, a_s1, a_s2, trainer_id):
        # Yizhi edit here
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.advantages: adv, self.v_targets: y_r})
        feed_dict.update({self.obs1: x1, self.actions1: a1, self.old_means1: a_m1, self.old_log_stds1: a_s1})
        feed_dict.update({self.obs2: x2, self.actions2: a2, self.old_means2: a_m2, self.old_log_stds2: a_s2,})
        self.sess.run(self.train_op_p1, feed_dict=feed_dict)
        self.sess.run(self.train_op_p2, feed_dict=feed_dict)
        self.sess.run(self.train_op_v, feed_dict=feed_dict)
        

    def log(self, x1, x2, y_r, adv, a1, a2, a_m1, a_m2, a_s1, a_s2):
        # Yizhi edit here
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.obs1: x1, self.obs2: x2, self.advantages:adv, self.actions1: a1, self.actions2: a2})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)
        

    def _checkpoint_filename(self, episode):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
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