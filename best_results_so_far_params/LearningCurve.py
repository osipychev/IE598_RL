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
                      "BETA_START:\s+{:.1f}\s+BETA_END:\s+{:.1f}.+"
                      "GAE_LAMBDA:\s+{:.1f}".format(beta_start, beta_end, gae_lambda))
            match = re.search(pattern, line)
            if match:
                time = float(match.group(1))
                rscore = float(match.group(2))
                yield time, rscore

job_id = "7895868"
lambdas = [1.0, 0.9, 0]

sns.set()
sns.set_palette("husl", 3)
plt.ion()

fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
for gae_lambda in lambdas:
    times, scores = zip(*read_scores(job_id, 0.0, 1.0, gae_lambda))
    times = np.asarray(times, np.float32)
    scores = np.asarray(scores, np.float32)
    ax.plot(times/3600, scores, 
                label=r"$\beta_s$={:g}, $\beta_e$={:g}, $\lambda$={:g}".format(0.0, 1.0, gae_lambda))
ax.legend()
ax.set_xlabel("Time (Hours)")
ax.set_ylabel("RScore")
ax.set_xlim([-0.2, 12])
plt.savefig("learning_curve.pdf", bbox_inches="tight")
