# Multi-Agent Reinforcement Learning project (MARL)

## Lunar Lander (continuous action space)
See details in folder [lunarlander_continuous](https://github.com/osipychev/IE598_RL/tree/master/lunarlander_continuous).

## Gathering (discrete action space)
See details in branch [gathering](https://github.com/osipychev/IE598_RL/tree/gathering).

## MARL Lunar Lander
I added GYM folder that is a clone of the original GYM with some modification.
The simplest way to make it working, just put https://github.com/osipychev/IE598_RL/tree/master/gym/gym
into your working folder like "workdir/gym/env".
The other way is to install gym from that repo (into a separate conda environment).
If installation to a separate conda env or virtual machine is impossible, the changes can be done to the original GYM installed to the system.
For that, you need to find the location of GYM that you run from and copy the swap ENV folder.

## Things that we've tried so far
1. We have modified the original lunar lander environment in OpenAI Gym to be able to take 2 agents such that we could use it as multi-agent environment for continuous action space.
2. The environment has been tested to be working fine.
3. Unfortunately, we haven't successfully trained the environment with any of the following three models:
   * Simple modified policy gradient codes given by Professor to train multiple agents.
   * Original homework 7 codes with doubled action and oberservation spaces.
   * Modified homework 7 codes such that the each agent has its own observations, policy and actions but share the same critic.
4. The training process wourld start but the RScores wouldn't increase and sometimes froze after some certain episodes.
