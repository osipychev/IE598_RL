# Multi-Agent Reinforcement Learning project (MARL)

## Food Gathering

This is a discrete action space problem with a gathering agents. The code is based on homework 6.

The environment is based on [MAS_environment](https://github.com/wwxFromTju/deepmind_MAS_enviroment/blob/master/MAS_enviroment/MAS_Gathering.py). The environment is already 2-agent but to be consistent with original code in homework 6 we still need to modify its GameEnv.move() function in `GameManager.py` as well as related files `Environment.py` and `ProcessAgent.py` (search the comment **need to mofidy the environment**).

Each agent has action and reward but the state/observation is the entire image so both agents share the same observation and value (unlike the continuous case). The system reward is the sum of their own rewards. Thus they also share the same advantage and critic function.

The modifications are made in `Config.py`, `NetworkVP.py`, `Server.py`, `ProcessAgent.py`, `ThreadTrainer.py`, `GameManager.py`, `Environment.py` and `Experiences.py` (search the comment **Yizhi edit here**).