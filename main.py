import torch
import datetime
import argparse
import numpy as np
import datetime
import sys, os

from envs.swat_gym import SWATEnv
from agents.random_agent import RandomAgent
from agents.standard_practice import StandardAgent
from agents.reactive_agent import ReactiveAgent
from agents.ddpg import DDPG, ReplayBuffer
from agents.td3 import TD3


# Runs policy for X episodes and returns average reward
def eval_policy(agent, max_action, seed, eval_episodes=10):
    eval_env = SWATEnv()
    # eval_env.seed(seed + 100)
    avg_reward = 0.

    for t in range(eval_episodes):
        state, _, done, info = eval_env.reset(seed=seed+100)
        while not done:
            action = agent.select_action(np.array(state)) 
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def train(env, action_dim, agent, memory, seed):
    batch_size = 8
    max_timesteps = env.max_duration+1 
    max_action = float(env.action_space.high[0])

    # init env
    state, _, done, info = env.reset()
    gaussian_std = 0.1 

    # bookkeeping
    rewards = []
    evaluations = []
    mu_loss = []
    q_loss = []
    for t in range(max_timesteps):
        # select action
        action = (
            agent.select_action(np.array(state))
            + np.random.normal(0, max_action * gaussian_std, size=action_dim)
        ).clip(0, max_action)

        # since max_action is set to fertilizer amount, implicitly clip max irrigation 
        # action = [action[0], action[1]/3.0]

        # perform action in env
        next_state, reward, done, info = env.step(action)

        # store experience in replay buffer 
        memory.add(state, action, next_state, reward, done)

        # update current state and date
        state = next_state

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
            df = env.show_history()
            state, _, done, info = env.reset()
            current_date = info[0]
            backup_rewards = rewards
            rewards = []
            # break

    rewards = backup_rewards
    return rewards, evaluations, df

# Runs policy for X episodes and returns average reward
def eval_random_policy(seed, eval_episodes=10):
    eval_env = SWATEnv()
    # eval_env.seed(seed + 100)
    avg_reward = 0.

    for t in range(eval_episodes):
        state, _, done, info = eval_env.reset(seed=seed+100)
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
    eval_env = SWATEnv()
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
    eval_env = SWATEnv()
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

def baseline_train(env, algorithm, agent, seed):
        # env.seed(seed + 100)
        max_timesteps = env.max_duration+1
        max_action = float(env.action_space.high[0])
        rewards = []
        evaluations = []

        # init env
        state, _, done, info = env.reset(seed=seed+100)
        current_date = datetime.datetime(2021, 4, 15)

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
                df = env.show_history()
                state, _, done, _ = env.reset()
                current_date = datetime.datetime(2021, 4, 15)
                backup_rewards = rewards
                rewards = []
                # break

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

            env = SWATEnv()
            # Set seeds
            # env.seed(seed)
            state, _, done, info = env.reset(seed=seed)
            env.action_space.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])
            print(f"state dim: {state_dim}, action_dim: {action_dim}, max action: {max_action}\n")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize policy, replayer buffer, and noise
            if algorithm=='Random':
                agent = RandomAgent()
                rewards, evals, df = baseline_train(env, algorithm, agent, seed)
            elif algorithm=='Standard':
                agent = StandardAgent(env.start_date)
                rewards, evals, df = baseline_train(env, algorithm, agent, seed)
            elif algorithm == 'Reactive':
                agent = ReactiveAgent()
                rewards, evals, df = baseline_train(env, algorithm, agent, seed)
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
                print("Incorrect policy specified. Valid choices = ['Standard', 'Reactive', 'DDPG']")
                sys.exit(0)

            file_name = f"{algorithm}_{seed}"
            savePath = f'./experiments/exp_{sim_time}/{file_name}'
            if not os.path.exists(savePath):
                os.makedirs(savePath)

            np.save(f"{savePath}/{file_name}_rewards", rewards)
            np.save(f"{savePath}/{file_name}_evals", evals)
            df.to_csv(f"{savePath}/{file_name}_history.csv")
    