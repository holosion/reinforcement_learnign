#loading up environment dependencies

import os
import pygame
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# Train without rendering so the training loop is not slowed by pygame.
log_path = os.path.join('Training', 'Logs')# create a path to save the logs
train_env = gym.make('CartPole-v1')

model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000, progress_bar=True)
model.save('Training/cartpole_ppo')
train_env.close()

# Evaluate the trained model in a fresh environment with a visible window.
eval_env = gym.make('CartPole-v1', render_mode='human')
for episode in range(1, 6):
    observation, info = eval_env.reset()
    done = False
    score = 0

    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        score += reward

    print('Evaluation episode {} score {}'.format(episode, score))
    
    
# SAVING OUR MODEL INTO OUR TRAINING FOLDER
PPO_Path = os.path.join('Training', 'savedmodel', 'PPO_Model_Cartpole')
model.save(PPO_Path)


#CARRYING OUT TESTS AND EVALUATIONS ON THE MODEL
evaluate_policy(model, train_env, n_eval_episodes = 10, render=True)

eval_env.close()



