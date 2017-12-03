#!/usr/bin/env python3

import re
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_scores(job_id):
    with open(job_id+".bw.log", "r") as f:
        for line in f:
            pattern = ("Time:\s+(\d+).+RScore:\s+(-?\d*\.\d+).+")
            match = re.search(pattern, line)
            if match:
                time = float(match.group(1))
                rscore = float(match.group(2))
                yield time, rscore

job_id = "7889067"

sns.set()
sns.set_palette("husl", 3)
plt.ion()

fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
times, scores = zip(*read_scores(job_id))
times = np.asarray(times, np.float32)
scores = np.asarray(scores, np.float32)
ax.plot(times/3600, scores, label=r"$\beta$={:g}, $\lambda$={:g}".format(0.01, 1.0))
ax.legend()
ax.set_xlabel("Time (Hours)")
ax.set_ylabel("RScore")
ax.set_xlim([-0.2, 12])
plt.savefig("learning_curve.pdf", bbox_inches="tight")
