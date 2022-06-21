# 2AMC15-DIC (Vacuum Cleaning Robot - FRIDAY)

## About this project
2AMC15 Data Intelligence Challenge Vacuum cleaning robot (Roomba) using Reinforcement Learning (RL). For this assignment, we 
created the environment with continuous action space (based on angle) using OpenAI Gym library and used PPO-clip RL algorithm
(hyper-parameters tuned) which is provided by Ray framework.

## How to run this app
You can run our project in two ways =>

1) Using Google Colab:
```
> Upload the complete A3_PPO folder in your drive
> Open the ppo_clip.ipynb file which is inside A3_PPO/algos folder from your drive.
> It will open the file in Google Colab environment. Change the path in Code block 3 to your A3_PPO folder
and then run (Runtime->Run all). It will start training the model for 200 epochs (dynamic environment at every epoch) and 
plot the losses and rewards. It will also save the trained model in checkpoints folder for further use. The code will then 
test the trained model for 10 epochs (again different dynamically changed env) and print the rewards and cleaning percentage
for 10 epochs.
```

2) On your local machine:

Install all required packages by running:
```
> pip install -r requirements.txt
> From the A3_PPO folder, run "pip install -e ." to setup directory structure
```

Run this app locally with (inside A3_PPO/algos folder):
```
> python ppo.py (inside A3_PPO/algos folder) - This will take default hyper-parameters (lr=1e-4 and gamma=0.99)
> python ppo.py --lr <learning-rate> --gamma <gamma> (inside A3_PPO/algos folder) - If you want to tune hyper-parameters
> python ppo.py --help (To get more info)
```

## Code Structure

    A3_PPO
    ├── algos                        # It contains all the algorithms
    │   ├── ppo.py                  # PPO agent
    │   ├── ppo_clip.ipynb          # PPO agent (Colab Notebook)
    │   └── random-agent.py         # Random agent
    ├── grids                      
    │   ├── house.grid             # Our house grid file
    ├── main                        
    │   ├── continuous.py            # create environment according to OpenAI gym format for our house grid
    │   ├── grid.py                  
    │   ├── parser.py              # parse the house.grid file
    │   ├── robot.py                # initialize robot
    │   ├── square.py               # get dirt area to calculate cleaning percentage
    ├── checkpoints
    ├── input.py                    # command-line input
    ├── setup.py                    # Set hierarchical directory structure
    ├── LICENSE                    
    ├── README.md                   # Readme file to explain about the project and how to run the code
    └── requirements.txt            # All the libraries and dependencies needed to run the project

## Resources

* [OpenAI Gym](https://www.gymlibrary.ml/content/environment_creation/)
* [PPO algorithm](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
* [Ray PPOtrainer](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#proximal-policy-optimization-ppo)
* [PPO Research Paper](https://arxiv.org/abs/1707.06347)
