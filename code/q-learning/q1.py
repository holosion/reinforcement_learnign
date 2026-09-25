# loading up environment dependencies
import os
import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# importing dependecies for applying the callbacks 
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold


# Create a path to save the logs
log_path = os.path.join('Training', 'Logs')
os.makedirs(log_path, exist_ok=True)

# Create a fresh environment for training
train_env = gym.make('CartPole-v1')

# Initialize the PPO model with TensorBoard logging
model = PPO(
    'MlpPolicy',
    train_env,
    verbose=1,
    tensorboard_log=log_path,
    device='cpu'
)

# Train the model
model.learn(total_timesteps=20000, progress_bar=True)
model.save('Training/cartpole_ppo')
train_env.close()

# Create a fresh environment for evaluation
eval_env = gym.make('CartPole-v1', render_mode='human')

# Evaluate the trained model
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
    
# Save the model to the specified path
PPO_Path = os.path.join('Training', 'savedmodel', 'PPO_Model_Cartpole')
model.save(PPO_Path)

# Evaluate the model using evaluate_policy
evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)

#setting up the callback
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(eval_env, callback_on_new_best=stop_callback, eval_freq=1000, best_model_save_path = PPO_Path)

eval_env.close()

# View the logs in TensorBoard
training_log_path = os.path.join(log_path, 'PPO_8')
print(f"TensorBoard logs are available at: {training_log_path}")
print("To view them, run: tensorboard --logdir={}".format(training_log_path))

