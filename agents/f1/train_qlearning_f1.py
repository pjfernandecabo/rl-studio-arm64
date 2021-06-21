
import datetime
import time
from cprint import cprint
from functools import reduce
import os

import gym

import numpy as np

import settings
import liveplot
import utils

from algorithms.qlearn import QLearn


def train_qlearning_f1(config):

    cprint.warn(f"\n {config['Title']}")
    cprint.ok(f"\n {config['Description']}")
    cprint.info(f"\n- Start hour: {datetime.datetime.now()}")


    #--------------------- Init QLearning Vars
    algorithm_hyperparams = config['Hyperparams']
    model = config['Model']

    # ------------------ Init env
    environment = {}
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

    print(f"\n environment: {environment}")

    env = gym.make(environment["env"], **environment)

    cprint.info(f"\n ---- come back train_qlearn_f1 ------------")

    # ---------------- Init hyperparmas & Algorithm
    alpha = config['Hyperparams']['alpha']
    gamma = config['Hyperparams']['gamma']
    initial_epsilon = config['Hyperparams']['epsilon']
    epsilon = config['Hyperparams']['epsilon']
    epsilon_discount = config['Hyperparams']['epsilon_discount']
    total_episodes = config['Hyperparams']['total_episodes']
    estimated_steps = config['envs_params'][model]['estimated_steps']

    actions = range(env.action_space.n) # lo recibe de F1QlearnCameraEnv
    qlearn = QLearn(actions=actions, alpha=alpha, gamma=gamma, epsilon=epsilon)    
    

    # ---------------- Init vars 
    outdir = f"{config['Dirspace']}/logs/{config['Method']}_{config['Algorithm']}_{config['Agent']}"
    #print(f"\n outdir: {outdir}")    
    os.makedirs(f"{outdir}", exist_ok=True)

    env = gym.wrappers.Monitor(env, outdir, force=True)

    stats = {}  # epoch: steps
    states_counter = {}
    states_reward = {}    
    last_time_steps = np.ndarray(0) 
    
    plotter = liveplot.LivePlot(outdir)
    
    counter = 0
    estimate_step_per_lap = environment["estimated_steps"]
    lap_completed = config['lap_completed']

    telemetry_start_time = time.time()
    start_time = datetime.datetime.now()
    start_time_format = start_time.strftime("%Y%m%d_%H%M")
    previous = datetime.datetime.now()
    checkpoints = []  # "ID" - x, y - time 

    if config['load_model']:
        # TODO: Folder to models. Maybe from environment variable?
        file_name = ''
        utils.load_model(qlearn, file_name)
        highest_reward = max(qlearn.q.values(), key=stats.get)
    else:
        highest_reward = 0

    cprint.warn(f"\n {config['Lets_go']}")


    ############################## START TRAINING #################################################################
    for episode in range(total_episodes):

        counter = 0
        done = False
        lap_completed = False

        cumulated_reward = 0
        observation = env.reset()

        if epsilon > 0.05:
            epsilon *= epsilon_discount

        state = ''.join(map(str, observation))
        print(f"\n -> START episode {episode}")
        print(f"counter: {counter}, observation: {observation}, done: {done}"
                f", lap_completed: {lap_completed}, cumulated_reward: {cumulated_reward}"
                f". state {state} with type {type(state)} in episode {episode}")


        for step in range(estimated_steps): #original range(500_000)

            counter += 1

            # Pick an action based on the current state
            action = qlearn.selectAction(state)
            print(f"\n ---- > START step {step}: action: {action} in episode {episode}")

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            print(f"\n In each step ==> observation: {observation}, reward: {reward}"
                  f", done: {done}, info: {info} in episode {episode}")
            
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            try:
                states_counter[nextState] += 1
            except KeyError:
                states_counter[nextState] = 1

            qlearn.learn(state, action, reward, nextState)

            env._flush(force=True)

            #if config['save_positions']:
            #    now = datetime.datetime.now()
            #    if now - datetime.timedelta(seconds=3) > previous:
            #        previous = datetime.datetime.now()
            #        x, y = env.get_position()
            #        checkpoints.append([len(checkpoints), (x, y), datetime.datetime.now().strftime('%M:%S.%f')[-4]])

            #    if datetime.datetime.now() - datetime.timedelta(minutes=3, seconds=12) > start_time:
            #        print("Finish. Saving parameters . . .")
            #        utils.save_times(checkpoints)
            #        env.close()
            #        exit(0)

            if not done:
                state = nextState
            else:
                last_time_steps = np.append(last_time_steps, [int(step + 1)])
                stats[int(episode)] = step
                states_reward[int(episode)] = cumulated_reward
                print(f"EP: {episode + 1} - epsilon: {round(qlearn.epsilon, 2)} - Reward: {cumulated_reward}"
                      f"- Time: {start_time_format} - Steps: {step}")
                break

            if step > estimate_step_per_lap and not lap_completed:
                lap_completed = True
                if config['plotter_graphic']:
                    plotter.plot_steps_vs_epoch(stats, save=True)
                utils.save_model(outdir, qlearn, start_time_format, stats, states_counter, states_reward)
                print(f"\n\n====> LAP COMPLETED in: {datetime.datetime.now() - start_time} - episode: {episode}"
                      f" - Cum. Reward: {cumulated_reward} <====\n\n")

            if counter > 100: #oroginal 1000
                if config['plotter_graphic']:
                    plotter.plot_steps_vs_epoch(stats, save=True)
                epsilon *= epsilon_discount
                utils.save_model(outdir, qlearn, start_time_format, episode, states_counter, states_reward)
                print(f"\t- epsilon: {round(qlearn.epsilon, 2)}\n\t- cum reward: {cumulated_reward}\n\t- dict_size: "
                      f"{len(qlearn.q)}\n\t- time: {datetime.datetime.now()-start_time}\n\t- steps: {step}\n")
                counter = 0

            if datetime.datetime.now() - datetime.timedelta(hours=2) > start_time:
                print(config['eop'])
                utils.save_model(outdir, qlearn, start_time_format, stats, states_counter, states_reward)
                print(f"    - N epoch:     {episode}")
                print(f"    - Model size:  {len(qlearn.q)}")
                print(f"    - Action set:  {settings.actions_set}")
                print(f"    - Epsilon:     {round(qlearn.epsilon, 2)}")
                print(f"    - Cum. reward: {cumulated_reward}")

                env.close()
                exit(0)

        if episode % 1 == 0 and config['plotter_graphic']:
            # plotter.plot(env)
            plotter.plot_steps_vs_epoch(stats)
            # plotter.full_plot(env, stats, 2)  # optional parameter = mode (0, 1, 2)

        if episode % 250 == 0 and config['save_model'] and episode > 1:  #OJO: episode % 250 originally
            print(f"\nSaving model . . .\n")
            utils.save_model(outdir, qlearn, start_time_format, stats, states_counter, states_reward)

        m, s = divmod(int(time.time() - telemetry_start_time), 60)
        h, m = divmod(m, 60)

    print("Total EP: {} - epsilon: {} - ep. discount: {} - Highest Reward: {}".format(
            total_episodes,
            initial_epsilon,
            epsilon_discount,
            highest_reward
        )
    )

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    #plotter.plot_steps_vs_epoch(stats, save=True)

    env.close()

