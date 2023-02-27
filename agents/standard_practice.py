"""
Applies fertilizer based on growth stage with 3 apps. of 60 kg/ha and 6 apps of 2 inches/ha
The seasonal average irrigation for (grain) corn is approximately 560 mm
Goal: have enough N in the soil early enough to maximize yield potential during early growth, and of applying all of the N in a way that results in less loss.
"""

import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from envs.swat_gym import SWATEnv


class StandardAgent(object):
    def __init__(self, start_date=None):
        self.start_date = start_date
    
    def select_action(self, current_date):
        operation_schedule = [self.start_date+datetime.timedelta(days=7), self.start_date+datetime.timedelta(days=25), 
        self.start_date+datetime.timedelta(days=60), self.start_date+datetime.timedelta(days=90)]

        # early stage applications
        if current_date == self.start_date+datetime.timedelta(days=1):
            fertilizer = 60
            irrig = 0
        elif current_date == self.start_date+datetime.timedelta(days=8):
            fertilizer = 0
            irrig = 25
        
        # mid-season applications
        elif current_date == self.start_date+datetime.timedelta(days=31):
            fertilizer = 60
            irrig = 0
        elif current_date == self.start_date+datetime.timedelta(days=38):
            fertilizer = 0
            irrig = 25
        
        # late-stage applications
        elif current_date == self.start_date+datetime.timedelta(days=61):
            fertilizer = 45
            irrig = 0
        elif current_date == self.start_date+datetime.timedelta(days=68):
            fertilizer = 0
            irrig = 15

        # final stage applications
        elif current_date == self.start_date+datetime.timedelta(days=100): 
            fertilizer = 15
            irrig = 10         
        
        # default action
        else:
            fertilizer = 0
            irrig = 0

        action = [fertilizer, irrig]
        return action

def eval_policy(policy, max_action, eval_episodes=10):
    eval_env = SWATEnv(max_action=60)
    avg_reward = 0.

    for t in range(eval_episodes):
        state, _, done, info = eval_env.reset()
        current_date = datetime.datetime(2021, 4, 15)
        while not done:
            action = policy.select_action(current_date)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            current_date += datetime.timedelta(days=1)

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def main():
    # init environment
    env = SWATEnv(max_action=60)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"state dim: {state_dim}, action_dim: {action_dim}, max action: {max_action}\n")

    state, _, done, info = env.reset()
    current_date = datetime.datetime(2021, 4, 15)
    max_timesteps = env.max_duration+1
    
    agent = StandardAgent(start_date=env.start_date)

    rewards = []
    avg_rwds = []

    for t in range(max_timesteps):
        action = agent.select_action(current_date)
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
            avg_reward = eval_policy(agent, max_action, eval_episodes=10)
            avg_rwds.append(avg_reward)
        
        if done:
            print("Episode terminated successfully!")
            df = env.show_history()
            state, _, done, _ = env.reset()
            current_date = datetime.datetime(2021, 4, 15)
            break

    df.to_csv('./results/standard_episode_1_data.csv', index=False)
    env.close()
    
    # average reward where each episode has a # timesteps = growing season
    plt.plot(rewards)
    plt.xlabel('timestep (day)')
    plt.ylabel('reward')
    np.save('./results/standard_agent_rewards.npy', np.array(rewards))
    plt.savefig('./results/standard_agent_rewards.png')

    plt.plot(avg_rwds)
    plt.xlabel('timestep (day)')
    plt.ylabel('avg reward')
    np.save('./results/standard_agent_avg_rewards.npy', np.array(avg_rwds))
    plt.savefig('./results/standard_agent_avg_rewards.png')
