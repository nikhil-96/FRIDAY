import sys
import getopt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import matplotlib.pyplot as plt
#
from main.continuous import FloorCleaning
from main.robot import Robot
from main.parsing import parse_config
from main.square import get_area
from input import input
#
# parent_path = Path(".").resolve().parent
grid = parse_config(Path("../grids") / "example.grid")

robot = Robot(init_position=(0, 8))

test_cleaning = []
test_reward = []

for epoch in tqdm(range(10)):
    env = FloorCleaning(dict(robot=robot, grid=grid))
    obs = env.reset()
    #env.render()
    running_reward = 0.0
    initial_dust_area = sum([get_area(patch) for patch in env.grid.dirt_places]) + \
                        sum([get_area(patch) for patch in env.grid.much_dirt_places]) + sum([get_area(patch) for patch in env.grid.reg_dirt])
    s = 0
    done = False
    while not done:
        move = env.action_space.sample()
        obs, reward, done, info = env.step(move)
        #env.render()
        #print(f"move: {move/(2*np.pi)*360}, reward: {reward}")
        running_reward += reward
        s += 1

    final_dust_area = sum([get_area(patch) for patch in env.grid.dirt_places]) + \
                      sum([get_area(patch) for patch in env.grid.much_dirt_places]) + sum([get_area(patch) for patch in env.grid.reg_dirt])

    cleaning_area = initial_dust_area - final_dust_area
    cleaning_per = (cleaning_area / initial_dust_area) * 100

    test_reward.append(running_reward)
    test_cleaning.append(cleaning_per)
    print(f"Epoch: {epoch} -- Reward: {running_reward} -- Cleaning %: {cleaning_per}")

print(test_reward)
print(test_cleaning)
print("Avg test reward", np.mean(test_reward))
print("Avg test cleaning %", np.mean(test_cleaning))