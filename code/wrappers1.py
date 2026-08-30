import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

#star with a complex observation space
env = gym.make("CarRacing-v3")
env.observation_space.shape #(96, 96, 3) # 96 x 96 RGB image

#wrap it into a flattened observation int to a 1D array
wrapped_env = FlattenObservation(env)
wrapped_env.observation_space.shape #(27648,) # 27648 flattened values
