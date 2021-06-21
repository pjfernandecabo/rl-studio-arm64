import datetime
import time
import sys
import argparse
from cprint import cprint
from functools import reduce

import gym
import numpy as np
import os

# next imports
# from agents.f1.settings
# from agents.f1.liveplot
# from agents.f1.utils
# from agents.f1.qlearn

# Importing app local files
import settings
import liveplot
import utils
#import settings

#from algorithms.qlearn import QLearn
from agents.f1.train_qlearning_f1 import train_qlearning_f1



if __name__ == '__main__':

    # ------------------- Parameter parsing from YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()

    #config, algorithm1, algorithm_hyperparams1, model1, actions1, gaz_pos1 = utils.read_config(args.config_file)
    config = utils.read_config(args.config_file)

    execute_algor = f"{config['Method']}_{config['Algorithm']}_{config['Agent']}"
    print(f"\n [RLStudio] -> execute ALGORITHM : {execute_algor}")

    # ------------------- CREATE DIRS
    os.makedirs("logs", exist_ok=True)
    os.makedirs("stats", exist_ok=True)
    #os.makedirs("logs", exist_ok=True)
    


    if execute_algor == 'train_qlearning_f1':
        train_qlearning_f1(config)
        




