from main.continuous import FloorCleaning
from main.robot import Robot
from main.parsing import parse_config
import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

grid = parse_config("../grids/example.grid")
robot = Robot(init_position=(0, 8))

env = FloorCleaning(dict(robot=robot, grid=grid))
model = PPO("MultiInputPolicy", env, learning_rate=0.001, n_steps=2048, batch_size=32, n_epochs=100, gamma=0.99)
model = model.learn(total_timesteps=25000, eval_env=env, n_eval_episodes=10)
for i in tqdm(range(20)):
    #model = model.learn(total_timesteps=25000, eval_env=env, n_eval_episodes=10)
    model.train()

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    print(f"move: {action / (2 * np.pi) * 360}, reward: {rewards}")

    if done:
        print(f"Game over: {info['reason']}")
        break