#loading up environment dependencies

import os
import pygame
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# loading up the environment
env = gym.make('CartPole-v1', render_mode='human')
episodes = 5
for episode in range(1, episodes + 1):
    state, info = env.reset()# reset the environment to start a new episode
    done = False
    score = 0

    while not done: # while the episode is not done, keep taking actions
        action = env.action_space.sample() # take a random action from the action space
        n_state, reward, terminated, truncated, info = env.step(action) # take a step in the environment using the action
        done = terminated or truncated
        score += reward
    print('Episode {} score {}'.format(episode, score))

env.close()

env.observation_space
env.observation_space.sample() # sample a random observation from the observation space

#traing the model using ppo algorithm
log_path = os.path.join('Training', 'Logs')# create a path to save the logs
env = DummyVecEnv([lambda: env])# wrap the environment in a DummyVecEnv to make it compatible with stable-baselines3

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000) #train the model for 20000 timesteps



