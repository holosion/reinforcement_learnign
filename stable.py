import gymnasium as gym

from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode="human")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    actions, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(actions)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()