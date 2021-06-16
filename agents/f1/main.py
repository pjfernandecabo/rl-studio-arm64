import datetime
import time
import sys
import argparse
from cprint import cprint
from functools import reduce

import gym
import numpy as np

# next imports
# from agents.f1.settings
# from agents.f1.liveplot
# from agents.f1.utils
# from agents.f1.qlearn
import settings
import liveplot
import utils
import qlearn
import train_qlearning_pedro





if __name__ == '__main__':

    # Parameter parsing from YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()

    config, algorithm = utils.read_config(args.config_file)

    print(f"\n params.yml: {config}")
    print(f"\n algorithm: {algorithm}")
    print(f"\n config['Hyperparams']: {config['Hyperparams']}")
    print(f"\n config['Hyperparams']['alpha']: {config['Hyperparams']['alpha']}")

    ## Init params 

    alpha = config['Hyperparams']['alpha']
    print(f"alpha: {alpha}")

    algorithm = f"{config['Method']}_{config['Algorithm']}_pedro"
    print(f"algorithm: {algorithm}")


    if algorithm == 'train_qlearning_pedro':
    	train_qlearning_pedro.train_qlearning_pedro(config)




