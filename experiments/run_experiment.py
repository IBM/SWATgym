"Run experiment evalauting all agents on SWATGym"

import torch
import datetime
import argparse
import numpy as np
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from envs.swat_gym import SWATEnv
from agents.ddpg import DDPG, ReplayBuffer
from agents.random_agent import RandomAgent
from agents.standard_practice import StandardAgent
from agents.reactive_agent import ReactiveAgent
from agents.td3 import TD3


# Runs policy for X episodes and returns average reward
def eval_policy(agent, max_action, seed, eval_episodes=10):
    eval_env = SWATEnv(seed=seed+100)
    # eval_env.seed(seed + 100)
    avg_reward = 0.

    for t in range(eval_episodes):
        state, _, done, info = eval_env.reset(seed=seed+100)
        while not done:
            action = agent.select_action(np.array(state)).clip(0.0, max_action)
            act0 = action[0] if not np.isnan(action[0]) else 0.0 
            act1 = action[1] if not np.isnan(action[1]) else 0.0
            action = np.array([act0, act1])
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def train(env, action_dim, agent, memory, seed):
    batch_size = 32
    max_timesteps = 700 # env.max_duration+1
    max_action = float(env.action_space.high[0])

    # init env
    state, _, done, info = env.reset(seed=seed)
    gaussian_std = 0.1 # Std of Gaussian exploration noise

    # bookkeeping
    rewards = []
    evaluations = [eval_policy(agent, max_action, seed, eval_episodes=10)]
    mu_loss = []
    q_loss = []
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(max_timesteps):
        episode_timesteps += 1

        # select action
        action = (
            agent.select_action(np.array(state))
            + np.random.normal(0, max_action * gaussian_std, size=action_dim)
        ).clip(0.0, max_action)

        # since max_action is set to fertilizer amount, implicitly clip max irrigation
        act0 = action[0] if not np.isnan(action[0]) else 0.0 
        act1 = action[1] if not np.isnan(action[1]) else 0.0
        action = np.array([act0, act1])

        # perform action in env
        next_state, reward, done, info = env.step(action)
        if np.isnan(reward):
            print(" invalid reward")
            print('action: ', action)
            print('state: \n', state)
            print('next obs: \n', next_state)
            print(f"Timestep: {t}, reward: {reward}")
            print(info)
            sys.exit(0)

        # store experience in replay buffer 
        memory.add(state, action, next_state, reward, done)

        # update current state and date
        state = next_state
        episode_reward += reward

        # track rewards
        rewards.append(reward)
        print(f"Timestep: {t}, reward: {reward}")

        # train after collecting enough samples
        if memory.size > 2*batch_size:
            if algorithm=='TD3':
                agent.train(memory, batch_size)
            else:
                actor_loss, critic_loss = agent.train(memory, batch_size)
                mu_loss.append(actor_loss)
                q_loss.append(critic_loss)

        # evaluate every week
        if (t+1)%7==0:
            avg_reward = eval_policy(agent, max_action, seed, eval_episodes=10)
            evaluations.append(avg_reward)
        
        if done:
            print("Episode terminated successfully!")
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            df = env.show_history()
            state, _, done, info = env.reset(seed=seed)
            backup_rewards = rewards
            rewards = []
            episode_reward = 0 
            episode_timesteps = 0 
            episode_num += 1 
            current_date = info[0]
    
    rewards = backup_rewards
    return rewards, evaluations, df

# Runs policy for X episodes and returns average reward
def eval_random_policy(seed, eval_episodes=10):
    eval_env = SWATEnv(seed=seed+100)
    # eval_env.seed(seed + 100)
    avg_reward = 0.

    for t in range(eval_episodes):
        state, _, done, info = eval_env.reset()
        while not done:
            action = env.action_space.sample() 
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def standard_eval_policy(agent, max_action, seed, eval_episodes=10):
    eval_env = SWATEnv(seed=seed+100)
    # eval_env.seed(seed + 100)
    avg_reward = 0.

    for t in range(eval_episodes):
        state, _, done, info = eval_env.reset(seed=seed+100)
        current_date = datetime.datetime(2021, 4, 15)
        while not done:
            action = agent.select_action(current_date)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            current_date += datetime.timedelta(days=1)

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def reactive_eval_policy(agent, max_action, seed, eval_episodes=10):
    eval_env = SWATEnv(seed=seed+100)
    # eval_env.seed(seed + 100)
    avg_reward = 0.

    for t in range(eval_episodes):
        state, _, done, info = eval_env.reset(seed=seed+100)
        current_date = datetime.datetime(2021, 4, 15)
        while not done:
            action = agent.select_action(current_date, state, info)
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            current_date += datetime.timedelta(days=1)

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def baseline_evals(env, algorithm, agent, seed):
        # env.seed(seed + 100)
        max_timesteps = 700 # env.max_duration+1
        max_action = float(env.action_space.high[0])
        rewards = []
        if algorithm=='Standard':
            avg_reward = standard_eval_policy(agent, max_action, seed, eval_episodes=10)
        elif algorithm=='Reactive':
            avg_reward = reactive_eval_policy(agent, max_action, seed, eval_episodes=10)
        else:
            avg_reward = eval_random_policy(seed)
        evaluations = [avg_reward]

        # init env
        state, _, done, info = env.reset(seed=seed)
        current_date = datetime.datetime(2021, 4, 15)
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(max_timesteps):
            if algorithm=='Standard':
                action = agent.select_action(current_date)
            elif algorithm=='Reactive':
                action = agent.select_action(current_date, state, info)
            else: # random
                action = env.action_space.sample() 

            # perform action in env
            next_state, reward, done, info = env.step(action)
            
            # update current state and date
            state = next_state
            episode_reward += reward
            current_date += datetime.timedelta(days=1)
            
            # track rewards
            rewards.append(reward)
            print(f"Timestep: {t}, reward: {reward}")

            # evaluate every week
            if (t+1)%7==0:
                if algorithm=='Standard':
                    avg_reward = standard_eval_policy(agent, max_action, seed, eval_episodes=10)
                elif algorithm=='Reactive':
                    avg_reward = reactive_eval_policy(agent, max_action, seed, eval_episodes=10)
                else:
                    avg_reward = eval_random_policy(seed)

                evaluations.append(avg_reward)
            
            if done:
                print("Episode terminated successfully!")
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                df = env.show_history()
                state, _, done, _ = env.reset(seed=seed)
                backup_rewards = rewards
                rewards = []
                episode_reward = 0 
                episode_timesteps = 0 
                episode_num += 1 
                current_date = datetime.datetime(2021, 4, 15)
        
        rewards = backup_rewards
        return rewards, evaluations, df


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="DDPG")
    args = parser.parse_args()

    algos = ['Random', 'Standard', 'Reactive', 'DDPG', 'TD3']
    sim_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    for seed in range(5):
        for algorithm in algos:
            print("---------------------------------------")
            print(f"Policy: {algorithm}, Seed: {seed}")
            print("---------------------------------------")

            env = SWATEnv(seed=seed)

            # Set seeds
            # env.seed(seed)
            env.action_space.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"state dim: {state_dim}, action_dim: {action_dim}, max action: {max_action}\n")

            # Initialize policy, replayer buffer, and noise
            if algorithm=='Random':
                agent = RandomAgent(env.start_date)
                rewards, evals, df = baseline_evals(env, algorithm, agent, seed)
            elif algorithm=='Standard':
                agent = StandardAgent(env.start_date)
                rewards, evals, df = baseline_evals(env, algorithm, agent, seed)
            elif algorithm == 'Reactive':
                agent = ReactiveAgent()
                rewards, evals, df = baseline_evals(env, algorithm, agent, seed)
            elif algorithm=='DDPG':
                agent = DDPG(state_dim, action_dim, max_action, device)
                memory = ReplayBuffer(state_dim, action_dim, device)
                rewards, evals, df = train(env, action_dim, agent, memory, seed)
            elif algorithm=='TD3':
                policy_noise = 0.2*max_action # Noise added to target policy during critic update
                noise_clip = 0.5*max_action #  Range to clip target policy noise
                policy_freq = 2  # Frequency of delayed policy updates
                memory = ReplayBuffer(state_dim, action_dim, device)
                agent = TD3(state_dim, action_dim, max_action, policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq)
                rewards, evals, df = train(env, action_dim, agent, memory, seed)
            else:
                print("Incorrect policy specified. Valid choices = ['Standard', 'Reactive', 'DDPG', 'TD3']")
                sys.exit(0)

            file_name = f"{algorithm}_{seed}"
            savePath = f'./results/exp_{sim_time}/{file_name}'
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            
            np.save(f"{savePath}/{file_name}_rewards", rewards)
            np.save(f"{savePath}/{file_name}_evals", evals)
            df.to_csv(f"{savePath}/{file_name}_history.csv")
    