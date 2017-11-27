# Multi-Agent Reinforcement Learning project (MARL)

## Lunar Lander

This is a continous action space problem with 2 landing agents, each of which has its 2D action. The code is based on homework 7.

Each agent has its own observation, action, action mean, action std, reward and value. So we need to modify the original environment accordingly especially the step() and reset() function: see details in `GameManager.py`, `ProcessAgent.py` and `Environment.py` (search the comment **need to modify the environment**).

Since the 2 agent will share the rewards and values eventually. The system reward and value are the sum of their own rewards and values. Thus they also share the same advantage and critic function.

The modifications are made in `Server.py`, `NetworkVP.py`, `ThreadTrainer.py`, `ProcessAgent.py`, `Environment.py`, `GameManager.py` and `Experiences.py` (search the comment **Yizhi edit here**).
