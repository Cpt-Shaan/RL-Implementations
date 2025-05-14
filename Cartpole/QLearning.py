'''
Q-Learning for cartpole gymnasium via discretization of state space

State Space - Position of robot, Velocity of robot, Pole Angle, Angular Velocity of Pole
Dividing the spaces via np.linspace()
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

def run(is_training = True, render = False):
    env = gym.make('CartPole-v1', render_mode = 'human' if render else None)
    env.reset()

    # Discretize position, velocity, pole angle and angular velocity
    pos_space = np.linspace(-2.4,2.4,10)
    vel_space = np.linspace(-4,4,10)
    ang_space = np.linspace(-0.2095,0.2095,10)
    ang_vel_space = np.linspace(-4,4,10)

    if(is_training):
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
        # 11 x 11 x 11 x 11 x 2 - Q-Table
    else:
        # Load q-table from binary file stored in after training
        f = open('cartpole.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    
    # Hyperparameters
    alpha = 0.1 # Learning Rate
    gamma = 0.99 # Discount factor
    epsilon = 1 # 100% random actions initially
    decay_rate = 1e-5 # Decay rate for epsilon

    rewards_per_episode = list()

    i = 0 # Episode Count

    while(True):  # Training till mean reward of last 100 episodes < 1000
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False

        rewards = 0

        i += 1
        while(not terminated and rewards < 10000):
            if is_training and random.random() < epsilon:  # Random action
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])
            
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + alpha * (
                    reward + gamma * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :]) - q[state_p, state_v, state_a, state_av, action]
                )
            
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av= new_state_av

            rewards += reward

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

        if is_training and i % 100 == 0:
            print(f"Episode no : {i}, Reward : {reward}, Epsilon : {epsilon:0.2f}, Mean Rewards : {mean_rewards:0.2f}")

        if mean_rewards > 1000:
            break

        epsilon = max(epsilon - decay_rate, 0)
    
    env.close()

    if is_training:
        f = open('cartpole.pkl', 'wb')
        pickle.dump(q,f)
        f.close()

    mean_rewards = list()
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.plot(mean_rewards)
    plt.savefig(f'cartpole.png')


if __name__ == '__main__':
    run(is_training = True, render = False)
    run(is_training = False, render = True)


