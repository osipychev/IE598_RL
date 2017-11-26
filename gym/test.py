import gym
env = gym.make('LunarLanderContinuousMarl-v2')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
