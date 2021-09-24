import time
from tqdm import tqdm
from cprint import cprint
from functools import reduce
import os
import gym
import numpy as np

import settings
import liveplot
import utils
from icecream import ic
from datetime import datetime, timedelta
import tensorflow as tf

from algorithms.Approximate.DQN.DQNAgent import DQNAgent, ModifiedTensorBoard
from gym_gazebo.envs.gazebo_env import *


ic.enable()
#ic.disable()
#ic.configureOutput(prefix='Debug | ')
ic.configureOutput(prefix=f'{datetime.now()} | ')


def train_dqn_V1(config):

        #--------------------- Init QLearning Vars
    model = config['Model']

    # ------------------ Init env
    environment = {}
    environment['agent'] = config['Agent']
    environment['env'] = config['envs_params'][model]['env']
    environment['training_type'] = config['envs_params'][model]['training_type']
    environment['circuit_name'] = config['envs_params'][model]['circuit_name']
    environment['actions'] = settings.AVAILABLE_ACTIONS[settings.actions_set]
    environment['launch'] = config['envs_params'][model]['launch']
    environment['gaz_pos'] = settings.GAZEBO_POSITIONS[config['envs_params'][model]['circuit_name']]
    environment['start_pose'] = [settings.GAZEBO_POSITIONS[config['envs_params'][model]['circuit_name']][1][1], settings.GAZEBO_POSITIONS[config['envs_params'][model]['circuit_name']][1][2]]
    environment['alternate_pose'] = config['envs_params'][model]['alternate_pose']
    environment['estimated_steps'] = config['envs_params'][model]['estimated_steps']
    environment['sensor'] = config['envs_params'][model]['sensor']
    environment['rewards'] = config['Rewards']
    environment['ROS_MASTER_URI'] = config['ROS_MASTER_URI']
    environment['GAZEBO_MASTER_URI'] = config['GAZEBO_MASTER_URI']


    #print(f"\n [train_qlearning_f1] -> environment: {environment}")
    ic(environment)

    env = gym.make(environment["env"], **environment)

    #cprint.info(f"\n ---- [train_qlearning_f1] -> come back train_qlearn_f1 ------------")

    # ---------------- Init hyperparmas & Algorithm
    alpha = config['Hyperparams']['alpha']
    gamma = config['Hyperparams']['gamma']
    initial_epsilon = config['Hyperparams']['epsilon']
    epsilon = config['Hyperparams']['epsilon']
    epsilon_discount = config['Hyperparams']['epsilon_discount']
    total_episodes = config['Hyperparams']['total_episodes']
    estimated_steps = config['envs_params'][model]['estimated_steps']

    actions = env.actions # lo recibe de F1DQNCameraEnv
    action_space = env.action_space # lo recibe de F1DQNCameraEnv
    action_space_n = range(env.action_space.n) # lo recibe de F1DQNCameraEnv
    action_space_size = env.action_space.n # lo recibe de F1DQNCameraEnv
    observation_space_values = (config['height_image'], config['width_image'], 3)
    observation_dim = (config['height_image'], config['width_image'], 3)
    ic(actions)    
    ic(action_space)    
    ic(action_space_n)    
    ic(action_space_size)    
    ic(observation_space_values)    
    ic(observation_dim)    


    # ---------------- Init vars 
    outdir = f"{config['Dirspace']}/logs/{config['Method']}_{config['Algorithm']}_{config['Agent']}"
    #print(f"\n outdir: {outdir}")    
    os.makedirs(f"{outdir}", exist_ok=True)
    ic(outdir)

    #env = gym.wrappers.Monitor(env, outdir, force=True)


    '''
        START DQN Agent
    '''
    agent_dqn = DQNAgent(config, action_space_size, observation_space_values, outdir)
    ic(agent_dqn)


    '''
        Init vars STATS
    '''
    stats = {}  # epoch: steps
    states_counter = {}
    states_reward = {}    
    ep_rewards = []
    aggr_ep_rewards = {
        'episode':[], 'avg':[], 'max':[], 'min':[], 'step':[], 'epsilon':[], 'time_training':[]
    } 
    last_time_steps = np.ndarray(0) 

    # For stats
    #ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.compat.v1.random.set_random_seed(1)



    #plotter = liveplot.LivePlot(outdir)
    
    counter = 0
    estimate_step_per_lap = environment["estimated_steps"]
    lap_completed = config['lap_completed']

    '''
        TIME
    '''
    start_time_training = time.time()
    telemetry_start_time = time.time()
    start_time = datetime.now()
    start_time_format = start_time.strftime("%Y%m%d_%H%M")
    previous = datetime.now()
    checkpoints = []  # "ID" - x, y - time 

    #cprint.warn(f"\n[train_qlearning_f1] -> {config['Lets_go']}")
    ic(config['Lets_go'])



    '''
        START TRAINING GOOOOOOOOOOO
    
    '''

    #start_time_training = time.time()
    for episode in tqdm(range(total_episodes), ascii=True, unit='episodes'):

        agent_dqn.tensorboard.step = episode

        counter = 0
        done = False
        lap_completed = False
        cumulated_reward = 0

        observation = env.reset()

        if epsilon > 0.05:
            epsilon *= epsilon_discount

        state = ''.join(map(str, observation))
        #print(f"\n -> START episode {episode}, counter: {counter}, observation: {observation}"
        #    f", done: {done}, lap_completed: {lap_completed}, cumulated_reward: {cumulated_reward}"
        #    f". state: {state}")

        #ic('START EPISODE')
        #ic(episode)    
        #ic(counter)    
        #ic(observation)    
        #ic(lap_completed)    
        #ic(cumulated_reward)    
        #ic(done)    
        #ic(state)  

        '''
            START STEPS in every episode
        '''

        for step in tqdm(range(estimated_steps)): #original range(500_000)

            counter += 1

            if np.random.random() > epsilon:
                # Get action from Q table
                #action = np.argmax(agent_dqn.get_qs(state))
                action = np.argmax(agent_dqn.get_qs(observation))
            else:
                # Get random action
                action = np.random.randint(0, action_space_size)

            ic(action)

            new_observation, reward, done, info = env.step(action)
            #print(f"\n ==> episode {episode}, step {step}: action: {action}"
            #    f", observation: {observation}, reward: {reward}"
            #    f", done: {done}, info: {info}")
            #ic('init step')
            #ic(episode)    
            #ic(step)    
            #ic(action)    
            #ic(new_observation)    
            #ic(reward)    
            #ic(done)    
            #ic(info)    
            
            cumulated_reward += reward

            #if highest_reward < cumulated_reward:
            #    highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            '''
            try:
                states_counter[nextState] += 1
            except KeyError:
                states_counter[nextState] = 1
            '''

            # Every step we update replay memory and train main network
            #agent_dqn.update_replay_memory((state, action, reward, nextState, done))
            agent_dqn.update_replay_memory((observation, action, reward, new_observation, done))
            agent_dqn.train(done, step)

            if not done:
                state = nextState
                observation = new_observation
            else:
                last_time_steps = np.append(last_time_steps, [int(step + 1)])
                stats[int(episode)] = step
                states_reward[int(episode)] = cumulated_reward
                print(f"Finish in: ---> Episode: {episode + 1} - Steps: {step} - epsilon: {epsilon}"
                      f"- Reward: {cumulated_reward} - Time: {start_time_format}")
                break


            # every 100 steps
            #if counter >= config['save_every_step']: #original 1000
            if not step % config['save_every_step'] and config['save_model']:
                epsilon *= epsilon_discount
                # utils.save_model(outdir, qlearn, start_time_format, episode, states_counter, states_reward)
                print(f"\tSAVING MODEL in step {step} in episode {episode}--------> \n")
                print(f"    - N epoch:     {episode}")
                #print(f"    - Model size:  {len(qlearn.q)}")
                print(f"    - Action set:  {settings.actions_set}")
                print(f"    - Epsilon:     {round(epsilon, 2)}")
                print(f"    - Cum. reward: {cumulated_reward}")      
                print(f"    - steps: {step}")      
                print(f"    - time: {datetime.now()-start_time}\n\t")   

                ic(episode)    
                ic(step)    
                #ic(counter)    
                ic(action)    
                ic(new_observation)    
                ic(reward)    
                ic(done)    
                ic(info)   

                #counter = 0



    env.close()


    '''
       ################################ BEGIN STATS ZONE
    '''
    end_time_training = time.time()
    training_time = end_time_training - start_time_training
    #print(f"\n ========= FINISH TRAINING at episode {episode} in time: {datetime.now()-start_time} ========\n\t")
    ic("========= FINISH TRAINING at episode")
    ic(episode)
    ic("in time:")
    ic(datetime.now()-start_time)
    ic(start_time_training)
    ic(end_time_training)
    ic(training_time)
    #print(f"Start Time: {start_time_training}, end time: {end_time_training}, and exec time: {training_time} in seconds")

    #print(f"Total Episodes: {total_episodes} - initial epsilon: {initial_epsilon} "
    #        f"- epsilon discount: {epsilon_discount} - Highest Reward: {highest_reward}")

    ic(total_episodes)
    ic(initial_epsilon)
    ic(epsilon_discount)
    #ic(highest_reward)

    # print("Parameters: a="+str)

    if done:
        print(f" ----- OVERALL STATS")
        print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    
        l = last_time_steps.tolist()
        l.sort()
        print("Best 100 scores: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

        #plotter.plot_steps_vs_epoch(stats, save=True)

