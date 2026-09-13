#loading up environment dependencies

import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# loading up the environment
env = gym.make('CartPole-v1')
episodes = 5
for episode in range(1, episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
        print('Episode{} score{}'.format(episode,score))



