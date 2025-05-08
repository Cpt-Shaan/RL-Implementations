# Q-Learning for frozen-lake

# some changes with respect to new gym version
# 1. need to change np.bool8 to np.bool in \Lib\site-packages\gym\utils\passive_env_checker.py due to newer version of numpy
# 2. environment.step(action) returns 5 params now (state,reward,done,info,prob)

import gym
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "rgb_array")
environment.reset()
environment.render()

"""
Our frozen lake environment has a total of 16 states (4x4 grid), and has 4 possible action for each state
Hence our q-table will consist of 16 rows and 4 cols - i.e 64 values, all initialized to 0
"""

"""
Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
"""

qtable = np.zeros((16,4))

# Alternatively can do via the gym library
states = environment.observation_space.n
actions = environment.action_space.n

qtable = np.zeros((states,actions))

# Hyperparameters
episodes = 1000        # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor

# List of outcomes to plot
outcomes = []

print('Q-table before training:')
print(qtable)

# Training process
for _ in range(episodes):
    state = environment.reset()[0]
    done = False

    # Consider outcome as failure by default
    outcomes.append("Failure")

    # continue steps till agent does not reach goal or gets stuck in hole
    while(not done):
        
        # chose action with highest value for gicen state
        if(np.max(qtable[state]) > 0):
            action = np.argmax(qtable[state])
        
        else: # go for a random action
            action = environment.action_space.sample()
        
        # take a step in the environment using this action
        new_state, reward, done, info, prob = environment.step(action)

        # update the Q-table as per the Q-Learning update equation
        qtable[state,action] = qtable[state,action] + \
                               alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state,action])
        
        state = new_state

        if reward:
            outcomes[-1] = "Success"

print()
print("Q-Table after training - ")
print(qtable)

# Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()

nb_success = 0

# Render for visualization
environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "human")
environment.reset()
environment.render()

# Evaluating for 100 episodes
for _ in range(100):
    state = environment.reset()[0]
    done = False
    
    while not done:
        
        if np.max(qtable[state]) > 0:
          action = np.argmax(qtable[state])

        else:
          action = environment.action_space.sample()
             
        new_state, reward, done, info, prob = environment.step(action)

        state = new_state

        nb_success += reward

print(f"Success rate = {nb_success/100*100}%")

