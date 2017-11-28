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

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from Config import Config
from Environment import Environment
from Experience import Experience


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.env = Environment()

        self.discount_factor = Config.DISCOUNT
        self.gae_lambda = Config.GAE_LAMBDA
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, gae_lambda, done):

      if done:
        experiences.append(Experience(None, None, None, None, None, None, None, None, 0, None, None))

      reward_sum = experiences[-1].value
      delta_sum = 0 
      for t in reversed(range(0, len(experiences)-1)):
        r = experiences[t].reward
        experiences[t].delta = r + discount_factor * experiences[t+1].value - experiences[t].value # delta (TD)

        reward_sum = discount_factor * reward_sum + r 
        experiences[t].reward = reward_sum # value target

        delta_sum = discount_factor * gae_lambda * delta_sum  + experiences[t].delta
        experiences[t].advantage = delta_sum # advantage
    
      return experiences[:-1]

    def convert_data(self, experiences):
        x1_ = np.array([exp.state1 for exp in experiences])
        x2_ = np.array([exp.state2 for exp in experiences])
        r_ = np.array([exp.reward for exp in experiences])
        adv_ = np.array([exp.advantage for exp in experiences])
        a1_ = np.array([exp.action1 for exp in experiences])
        a2_ = np.array([exp.action2 for exp in experiences])
        am1_ = np.array([exp.action_mean1 for exp in experiences])
        am2_ = np.array([exp.action_mean2 for exp in experiences])
        as1_ = np.array([exp.action_log_std1 for exp in experiences])
        as2_ = np.array([exp.action_log_std2 for exp in experiences])
        return x1_, x2_, r_, adv_, a1_, a2_, am1_, am2_, as1_, as2_
    
    def predict(self, state1, state2):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state1, state2))
        # wait for the prediction to come back
        mean1, mean2, log_std1, log_std2, value = self.wait_q.get()
        return mean1, mean2, log_std1, log_std2, value
        #return mean, log_std, value[0]

    def select_action(self, mean, log_std):
        if Config.PLAY_MODE:
            action = mean
        else:                   
            rnd = np.random.normal(size=mean.shape)
            action = rnd * np.exp(log_std) + mean
        return action

    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        while not done:
            # Yizhi edit here
            # agent 1
            obs1 = self.env.current_state1
            #action_mean1, action_log_std1, value1 = self.predict(obs1, obs2)
            #action1 = self.select_action(action_mean1, action_log_std1)

            # agent 2
            obs2 = self.env.current_state2
            action_mean1, action_mean2, action_log_std1, action_log_std2, value = self.predict(obs1,obs2)
            action1 = self.select_action(action_mean1, action_log_std1)
            action2 = self.select_action(action_mean2, action_log_std2)

            # need to modify the environment
            reward1, reward2, done = self.env.step(action1, action2)
            reward = reward1 + reward2
            reward_sum += reward
            exp = Experience(obs1, obs2, action1, action2, action_mean1, action_mean2, 
                                action_log_std1, action_log_std2, value, reward, done)
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:

                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, self.gae_lambda, done)
                x1_, x2_, r_, adv_, a1_, a2_, am1_, am2_, as1_, as2_ = self.convert_data(updated_exps)
                yield x1_, x2_, r_, adv_, a1_, a2_, am1_, am2_, as1_, as2_, reward_sum

                # reset the tmax count
                time_count = 0
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            # Yizhi edit here
            for x1_, x2_, r_, adv_, a1_, a2_, am1_, am2_, as1_, as2_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x1_, x2_, r_, adv_, a1_, a2_, am1_, am2_, as1_, as2_))
            self.episode_log_q.put((datetime.now(), total_reward, total_length))
