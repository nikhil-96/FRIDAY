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

lr, gamma = input(sys.argv)
robot = Robot(init_position=(0, 8))
ray.shutdown()
ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 1
config["num_workers"] = 2
config["env_config"] = {"robot": robot, "grid": grid}
config["recreate_failed_workers"] = True
config["gamma"] = gamma
config["lr"] = lr

PPO_trainer = ppo.PPOTrainer(env=FloorCleaning, config=config)

checkpoint_path = f"../checkpoints/checkpoint{str(config['lr']).replace('.','')}_{str(config['gamma']).replace('.','')}"
train_losses = []
train_rewards = []
# trainer.train()
for i in tqdm(range(200)):
    # Perform one iteration of training the policy with PPO
    result = PPO_trainer.train()
    # print(result))
    train_rewards.append(result["episode_reward_mean"])
    train_losses.append(result["info"]["learner"]["default_policy"]["learner_stats"]["total_loss"])

checkpoint_path = PPO_trainer.save(checkpoint_dir=checkpoint_path)
print("checkpoint saved at", checkpoint_path)
plt.xlabel('train_epoch')
plt.ylabel('rewards')
plt.plot(train_rewards, label='Rewards')
plt.title(f"Rewards vs Epoch During Training (lr={config['lr']}, gamma={config['gamma']})")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()

plt.xlabel('train_epoch')
plt.ylabel('losses')
plt.plot(train_losses, label='Losses')
plt.title(f"Losses vs Epoch During Training (lr={config['lr']}, gamma={config['gamma']})")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()

np.savetxt(f"train_rewards{str(config['lr']).replace('.','')}_{str(config['gamma']).replace('.','')}.csv",
           train_rewards,
           delimiter =", ",
           fmt ='% s')

np.savetxt(f"train_losses{str(config['lr']).replace('.','')}_{str(config['gamma']).replace('.','')}.csv",
           train_losses,
           delimiter =", ",
           fmt ='% s')

#PPO_trainer.restore(f'{checkpoint_path}/checkpoint_000300/checkpoint-300')
PPO_trainer.restore(checkpoint_path)

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
        move = PPO_trainer.compute_action(obs)
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
