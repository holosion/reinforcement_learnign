import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt

# Create the environment
env = gym.make("CarRacing-v3")

# Reset the environment and get the original image
observation, info = env.reset()

print("Original observation shape:", observation.shape)

# Display the image
plt.imshow(observation)
plt.axis("off")
plt.show()

# Now flatten the observation
wrapped_env = FlattenObservation(env)

flattened_observation, info = wrapped_env.reset()

print("Flattened observation shape:", flattened_observation.shape)
print("Number of values:", flattened_observation.size)
print("First 10 values:", flattened_observation[:10])

wrapped_env.close()