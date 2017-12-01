#!/usr/bin/env python3

import re
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_scores(job_id, beta_start, beta_end, gae_lambda):
    with open(job_id+".bw.log", "r") as f:
        for line in f:
            pattern = ("Time:\s+(\d+).+RScore:\s+(-?\d*\.\d+).+"
                      "BETA_START:\s+{:.2f}\s+BETA_END:\s+{:.2f}.+"
                      "GAE_LAMBDA:\s+{:.2f}".format(beta_start, beta_end, gae_lambda))
            match = re.search(pattern, line)
            if match:
                time = float(match.group(1))
                rscore = float(match.group(2))
                yield time, rscore

job_id = "7888982"
game_name = "LunarLanderContinousMarl"

betas = [1.0, 0, 0.01, 0.01]
lambdas = [1.0, 1.0, 0.9, 0]

sns.set()
sns.set_palette("husl", 4)
plt.ion()

fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
for beta, gae_lambda in zip(betas, lambdas):
    times, scores = zip(*read_scores(job_id, beta, beta, gae_lambda))
    times = np.asarray(times, np.float32)
    scores = np.asarray(scores, np.float32)
    ax.plot(times/3600, scores, 
                label=r"$\beta$={:g}, $\lambda$={:g}".format(beta, gae_lambda))
ax.legend()
ax.set_xlabel("Time (Hours)")
ax.set_ylabel("RScore")
ax.set_xlim([-0.2, 12])
plt.savefig("../../results_for_report/learning_curve_betas_lambdas.pdf", bbox_inches="tight")
