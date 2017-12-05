import gym
import numpy as np
#env = gym.make('LunarLanderMarl-v2')
#env.reset()
#for _ in range(100):
#    env.render()
#    action = env.action_space.sample()
#    #print("action: %d" % action)
#    env.step(action)


env = gym.make('LunarLanderContinuousMarl-v2')

for j in range(100):
    env.reset()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        #print("action: %d" % action)
        obs, rew, don, info = env.step(np.array(action))
        print(rew)        
