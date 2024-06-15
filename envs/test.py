import numpy as np
from rlss_envs import ServerlessEnv as env
import gymnasium as gym

# Set the parameters for the Q-learning
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.99
min_epsilon = 0.01
episodes = 10000
max_steps = 100

# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Q-learning algorithm
for episode in range(episodes):
    state = env.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        step += 1
        
        # Choose action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
        
        # Take action and observe the result
        next_state, reward, done, _ = env.step(action)
        
        # Update Q-table
        q_value = q_table[state, action]
        max_next_q_value = np.max(q_table[next_state])
        q_table[state, action] = q_value + learning_rate * (reward + discount_factor * max_next_q_value - q_value)
        
        state = next_state
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training finished.\n")

# Test the agent
state = env.reset()
env.render()
done = False

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _ = env.step(action)
    env.render()

print("Test finished.\n")