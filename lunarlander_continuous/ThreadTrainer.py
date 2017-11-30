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

from threading import Thread
import numpy as np

from Config import Config


class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        while not self.exit_flag:
            batch_size = 0
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                # Yizhi edit here
                x1_, x2_, r_, adv_, a1_, a2_, am1_, am2_, as1_, as2_ = self.server.training_q.get()
                if batch_size == 0:
                    x1__ = x1_; x2__ = x2_; r__ = r_; adv__ = adv_; a1__ = a1_; a2__ = a2_
                    am1__ = am1_; am2__ = am2_; as1__ = as1_; as2__ = as2_ 
                else:
                    x1__ = np.concatenate((x1__, x1_))
                    x2__ = np.concatenate((x2__, x2_))
                    r__ = np.concatenate((r__, r_))
                    adv__ = np.concatenate((adv__, adv_))
                    a1__ = np.concatenate((a1__, a1_))
                    a2__ = np.concatenate((a2__, a2_))
                    am1__ = np.concatenate((am1__, am1_))
                    am2__ = np.concatenate((am2__, am2_))
                    as1__ = np.concatenate((as1__, as1_))
                    as2__ = np.concatenate((as2__, as2_))
                batch_size += x1_.shape[0]
            
            if Config.TRAIN_MODELS:
                self.server.train_model(x1__, x2__, r__, adv__, a1__, a2__, 
                                am1__, am2__, as1__, as2__, self.id)
