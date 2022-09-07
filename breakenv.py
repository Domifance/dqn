import gym
import numpy as np
import time
# env = gym.make("LunarLander-v2")
# env = gym.make("ALE/Breakout-v5")
env = gym.make("Breakout-v4")
# env.action_space.seed(42)

observation = env.reset()
# print(observation,observation.shape)
# print(env.action_space)
env.render()
for _ in range(1000):
    action = env.action_space.sample()
    print(action)
    observation, reward, done, info = env.step(action)
    # print(observation)
    print(np.sum(observation))
    time.sleep(0.5)
    # a = input()
    print("done",done)

    if done:
        print("done:")
        a = input()
        # observation, info = env.reset()
        observation = env.reset()
        # print(observation)
        print(reward)

env.close()