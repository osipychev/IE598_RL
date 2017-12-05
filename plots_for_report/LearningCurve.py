#!/usr/bin/env python3

import re
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_scores(job):
    base = "../code_for_report"
    with open(join(base, job), "r") as f:
        for line in f:
            pattern = ("Episode:\s+(\d+).+RScore:\s+(-?\d*\.\d+).+")
            match = re.search(pattern, line)
            if match:
                episode = float(match.group(1))
                rscore = float(match.group(2))
                yield episode, rscore


def read_scores_params(job, beta_start, beta_end, gae_lambda):
    base = "../code_for_report"
    with open(join(base, job), "r") as f:
        for line in f:
            pattern = ("Episode:\s+(\d+).+RScore:\s+(-?\d*\.\d+).+"
                       "BETA_START:\s+{:.2f}\s+BETA_END:\s+{:.2f}.+"
                       "GAE_LAMBDA:\s+{:.2f}".format(beta_start, beta_end, gae_lambda))
            match = re.search(pattern, line)
            if match:
                episode = float(match.group(1))
                rscore = float(match.group(2))
                yield episode, rscore

# 2 actors vs 1 actor, old reward vs new reward
jobs = ["s3_modified_env_with_2actors/7889067.bw.log",
        "s4_modified_env_with_1pol_net/7889125.bw.log",
        "s6_best_results_64neur/result.log",
        "s9_single_actor/results.log"]
labels = ["2 actors, old reward", "1 actor, old reward",
          "2 actors, new reward", "1 actor, new reward"]

sns.set()
sns.set_palette("husl", 4)
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
for job, label in zip(jobs, labels):
    episode, score = zip(*read_scores(job))
    episode = np.asarray(episode, np.float32)
    score = np.asarray(score, np.float32)
    ax.plot(episode, score, label=label)
ax.legend()
ax.set_xlabel("Episode")
ax.set_ylabel("RScore")
ax.set_xlim([-3000, 60000])
ax.set_ylim([-1000, 0])
plt.savefig("actor_reward_comparison.pdf", bbox_inches="tight")
plt.savefig("actor_reward_comparison.png", bbox_inches="tight")

# lambdas and betas
jobs = ["s3_modified_env_with_2actors/7889067.bw.log",
        "s5_modified_env_with_2actors_params/7888982.bw.log"]
betas = [0.0, 1.0, 0.01, 0.01, 0.01]
lambdas = [1.0, 1.0, 1.0, 0.9, 0.0]

sns.set()
sns.set_palette("husl", 5)
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
for beta, gae_lambda in zip(betas, lambdas):
    if beta == 0.01 and gae_lambda == 1.0:
        episode, score = zip(*read_scores(jobs[0]))
    else:
        episode, score = zip(*read_scores_params(jobs[1], beta, beta, gae_lambda))
    episode = np.asarray(episode, np.float32)
    score = np.asarray(score, np.float32)
    ax.plot(episode, score, 
                label=r"$\beta$={:g}, $\lambda$={:g}".format(beta, gae_lambda))
ax.legend()
ax.set_xlabel("Episode")
ax.set_ylabel("RScore")
ax.set_xlim([-3000, 60000])
ax.set_ylim([-1500, 200])
plt.savefig("lambda_beta_comparison.pdf", bbox_inches="tight")
plt.savefig("lambda_beta_comparison.png", bbox_inches="tight")

# number of neurons
jobs = ["s6_best_results_64neur/result.log",
        "s7_new_env_with_32neur/results.log",
        "s8_new_env_with_16neur/results.log"]
labels = [64, 32, 16]

sns.set()
sns.set_palette("husl", 3)
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
for job, label in zip(jobs, labels):
    episode, score = zip(*read_scores(job))
    episode = np.asarray(episode, np.float32)
    score = np.asarray(score, np.float32)
    ax.plot(episode, score, label="{:d} neurons".format(label))
ax.legend()
ax.set_xlabel("Episode")
ax.set_ylabel("RScore")
ax.set_xlim([-3000, 60000])
ax.set_ylim([-1000, 0])
plt.savefig("neurons_comparison.pdf", bbox_inches="tight")
plt.savefig("neurons_comparison.png", bbox_inches="tight")