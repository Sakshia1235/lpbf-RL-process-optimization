import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
a1, b1, c1 = -5.33335E-05, 0.005571812, 2.321736449
a2, b2, c2 = -0.000164653, 0.143259306, -1.908263576

def random_argmax(array):
    max_value = np.max(array)
    max_indices = np.where(array == max_value)[0]
    return random.choice(max_indices)

# Ranges
v_range = np.arange(40, 101, 0.1) ##610 values in state space v
P_range = np.arange(75, 301, 0.1) #2260 values in state space P Total values are 610*2260= 13786000 in (p,V)

# Parameters
epsilon = 0.2
epsilon_decay = 0.9995
alpha = 0.01
gamma = 0.7
num_episodes = 50000
max_steps = 8

#  MODIFIED reward function
def reward_function(v, P):
    f = (a1 * v**2 + b1 * v + c1) * (a2 * P**2 + b2 * P + c2)
    if f > 0:
        return np.tanh(1 / f)  # Smoothed positive reward
    else:
        return -1 / (1 + abs(f))  #Scaled smooth penalty

episode_rewards = []

# Actions
actions = [
    ('increase_v', 'decrease_P'),
    ('increase_v', 'increase_P'),
    ('decrease_v', 'decrease_P'),
    ('decrease_v', 'increase_P'),
    ('increase_v', 'no_change_P'),
    ('decrease_v', 'no_change_P'),
    ('no_change_v', 'increase_P'),
    ('no_change_v', 'decrease_P'),
    ('no_change_v', 'no_change_P')
]

Q_table = np.ones((len(v_range), len(P_range), len(actions))) * 10.0 #Intialising Q table with high values

best_seen_reward = -float('inf')
best_state_from_training = None

break_var=0


# Q-learning
for episode in range(num_episodes):
    if break_var==0:
        v = random.choice(v_range) #Randomly select initial v
        P = random.choice(P_range) #Randomly select initial P
        total_reward = 0
        break_var+=1

    for step in range(max_steps):
        v_index = np.where(v_range == v)[0][0]  # Get index of current v
        P_index = np.where(P_range == P)[0][0]  # Get index of current P
        indexdeltav = 1
        indexdelta = 1

        if np.random.rand() < epsilon:
            action_index = random.choice(range(len(actions)))
        else:
            action_index = random_argmax(Q_table[v_index, P_index])

        action = actions[action_index]
        
        if v_index + indexdeltav <= 610 & v_index - indexdeltav >= 0:
            indexdeltav = int(reward/best_seen_reward) #index delta v should increase if the reward is large compared to best seen reward and decrease if opposite
        else:
            indexdeltav = 1
        
        if P_index + indexdelta <= 2260 & P_index - indexdelta >= 0:
            indexdelta = int(reward/best_seen_reward) #index delta p should increase if the reward is large compared to best seen reward and decrease if opposite
        else:
            indexdelta = 1
        
        if action == ('increase_v', 'decrease_P'):
            v = v_range[v_index + indexdeltav] if v_index + 1 < len(v_range) else v
            P = P_range[P_index - indexdelta] if P_index > 0 else P
        elif action == ('increase_v', 'increase_P'):
            v = v_range[v_index + indexdeltav] if v_index + 1 < len(v_range) else v
            P = P_range[P_index + indexdelta] if P_index + 1 < len(P_range) else P
        elif action == ('decrease_v', 'decrease_P'):
            v = v_range[v_index - indexdeltav] if v_index > 0 else v
            P = P_range[P_index - indexdelta] if P_index > 0 else P
        elif action == ('decrease_v', 'increase_P'):
            v = v_range[v_index - indexdeltav] if v_index > 0 else v
            P = P_range[P_index + indexdelta] if P_index + 1 < len(P_range) else P
        elif action == ('increase_v', 'no_change_P'):
            v = v_range[v_index + indexdeltav] if v_index + 1 < len(v_range) else v
        elif action == ('decrease_v', 'no_change_P'):
            v = v_range[v_index - indexdeltav] if v_index > 0 else v
        elif action == ('no_change_v', 'increase_P'):
            P = P_range[P_index + indexdelta] if P_index + 1 < len(P_range) else P
        elif action == ('no_change_v', 'decrease_P'):
            P = P_range[P_index - indexdelta] if P_index > 0 else P

        v = max(min(v, v_range[-1]), v_range[0])
        P = max(min(P, P_range[-1]), P_range[0])

        # EARLY TERMINATION + reward improvement
        reward = reward_function(v, P)
        total_reward += reward

        print(f"Episode: {episode}, Step: {step}, Action: {action}")
        print(f"Reward: {reward}")

        if reward > best_seen_reward:
            best_seen_reward = reward
            best_state_from_training = (v, P)
            best_action = actions[action_index]

        next_v_index = np.where(v_range == v)[0][0]
        next_P_index = np.where(P_range == P)[0][0]

        Q_table[v_index, P_index, action_index] += alpha * (
            reward + gamma * np.max(Q_table[next_v_index, next_P_index]) - Q_table[v_index, P_index, action_index]
        )

        # TERMINATE episode early if penalty state
        if reward < 0:
            break_var=1
            # Optional final update if terminating
            break

    epsilon = max(0.01, epsilon * epsilon_decay)
    episode_rewards.append(best_seen_reward)




# Plotting
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_rewards = moving_average(episode_rewards, window_size=50)

plt.figure(figsize=(7, 5))
plt.plot(episode_rewards, alpha=0.3, label='Raw Reward')
plt.plot(smoothed_rewards, label='Smoothed Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward vs Episodes')
plt.legend()
plt.show()


# Final printout 
print(f"Best reward: {best_seen_reward} at state: v={best_state_from_training[0]}, P={best_state_from_training[1]}")
