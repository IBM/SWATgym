"""
The reactive agent applies:
    - 120kg/ha every time the soil nitrogen content depletes below 2.5kg/ha
    - Irrigation scheduling using the soil water balance method
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


class ReactiveAgent(object):
    def __init__(self, start_date=None):
        self.start_date = start_date
    
    def select_action(self, day, state, info):
        current_nitrogen_level = info[-1]
        soil_water_content = info[-2]
        fertilizer = 0
        irrigation = 0

        if current_nitrogen_level < 10:
            fertilizer = 60
        if soil_water_content < 25:
            irrigation = 50
            
        action = [fertilizer, irrigation]

        return action

    def auto_fert(self, nitrogen_level):
        fert = 0
        return fert

    def auto_irrigate(self, swc, w_stress):
        # necessary when sol water content is less than wilting point or within 10% range of WP
        fc = 43.13 # mm
        stress_threshold = 0.9  # threshold that triggers irrigation
        if swc < fc - stress_threshold:
            irrigation = 1
        else:
            irrigation = 0
        return irrigation

def eval_policy(policy, max_action, eval_episodes=10):
    eval_env = SWATEnv(max_action=120)
    avg_reward = 0.

    for t in range(eval_episodes):
        state, _, done, info = eval_env.reset()
        current_date = datetime.datetime(2021, 4, 15)
        while not done:
            action = policy.select_action(current_date, state, info)
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            current_date += datetime.timedelta(days=1)

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def main():
    # init environment
    env = SWATEnv(max_action=120)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"state dim: {state_dim}, action_dim: {action_dim}, max action: {max_action}\n")

    state, _, done, info = env.reset()
    max_timesteps = env.max_duration+1
    current_date = datetime.datetime(2021, 4, 15)

    agent = ReactiveAgent()

    rewards = []
    avg_rwds = []

    for t in range(max_timesteps):
        action = agent.select_action(current_date, state, info)
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

    df.to_csv('./results/reactive_episode_1_data.csv', index=False)
    print(df.tail(10))
    env.close()

    # average reward where each episode has a # timesteps = growing season
    plt.plot(rewards)
    plt.xlabel('timestep (day)')
    plt.ylabel('reward')
    plt.savefig('./results/reactive_agent_rewards.png')
    np.save('./results/reactive_agent_rewards.npy', np.array(rewards))

    plt.plot(avg_rwds)
    plt.xlabel('timestep (day)')
    plt.ylabel('avg reward')
    np.save('./results/reactive_agent_avg_rewards.npy', np.array(avg_rwds))
    plt.savefig('./results/reactive_agent_avg_rewards.png')

if __name__=="__main__":
    main()
        
