import datetime
from envs.swat_gym import SWATEnv
from agents.random_agent import RandomAgent

def main():
    env = SWATEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"state dim: {state_dim}, action_dim: {action_dim}, max action: {max_action}\n")

    agent = RandomAgent()
    max_timesteps = env.max_duration+1
    rewards = []

    # init env
    state, _, done, info = env.reset()
    current_date = datetime.datetime(2021, 4, 15)

    for t in range(max_timesteps):
        # select action
        action = env.action_space.sample() 

        # perform action in env
        next_state, reward, done, info = env.step(action)
        
        # update current state and date
        state = next_state
        current_date += datetime.timedelta(days=1)
        
        # track rewards
        rewards.append(reward)
        print(f"Timestep: {t}, reward: {reward}")

        # check progress
        if done:
            print("Episode terminated successfully!")
            df = env.show_history()
            state, _, done, _ = env.reset()
            current_date = datetime.datetime(2021, 4, 15)
            rewards = []
            break

if __name__=="__main__":
    main()
