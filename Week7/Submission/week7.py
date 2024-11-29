import random
import matplotlib.pyplot as plt
import numpy as np
from operator import add

"""1. Binary Bandit implementation using an epsilon-greedy approach."""

class BanditA:
    def _init_(self):
        self.success_prob = 0.7
    
    def get_reward(self, selected_action):
        return 1 if random.random() < self.success_prob else 0

class BanditB:
    def _init_(self):
        self.success_prob = 0.3
    
    def get_reward(self, selected_action):
        return 1 if random.random() < self.success_prob else 0

def epsilon_greedy_strategy(bandit, eps, iterations):
    rewards = []
    avg_rewards = [0]
    estimated_q = 0
    total_actions = 0
    
    for step in range(1, iterations):
        if random.random() < eps:
            chosen_action = random.choice([0, 1])
        else:
            chosen_action = 1 if estimated_q > 0.5 else 0
        
        received_reward = bandit.get_reward(chosen_action)
        total_actions += 1
        estimated_q += (received_reward - estimated_q) / total_actions
        
        rewards.append(received_reward)
        avg_rewards.append(avg_rewards[step - 1] + (received_reward - avg_rewards[step - 1]) / step)
    
    plt.plot(range(1, iterations + 1), avg_rewards, label=f'epsilon = {eps}')
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()

# Testing Bandit A
bandit_a = BanditA()
plt.title('Binary Bandit A')
epsilon_greedy_strategy(bandit_a, 0.01, 10000)

# Testing Bandit B
bandit_b = BanditB()
plt.title('Binary Bandit B')
epsilon_greedy_strategy(bandit_b, 0.01, 10000)

"""2. Implementation of a 10-armed bandit with non-stationary rewards."""

class MultiArmBandit:
    def _init_(self, arms):
        self.num_arms = arms
        self.reward_means = [1] * arms

    def available_actions(self):
        return list(range(self.num_arms))

    def get_reward(self, chosen_arm):
        probabilities = [0.1 * i for i in range(self.num_arms)]
        return 1 if random.random() < probabilities[chosen_arm] else 0

    def non_stationary_reward(self, chosen_arm):
        noise = np.random.normal(0, 0.01, self.num_arms)
        self.reward_means = list(map(add, self.reward_means, noise))
        return self.reward_means[chosen_arm]

def epsilon_greedy_agent(multi_bandit, eps, max_steps=1000, strategy=1):
    est_values = [0] * multi_bandit.num_arms
    action_counts = [0] * multi_bandit.num_arms
    rewards_collected = []
    avg_rewards = [0]
    
    for step in range(1, max_steps):
        if random.random() > eps:
            selected_action = est_values.index(max(est_values))
        else:
            selected_action = random.choice(multi_bandit.available_actions())
        
        if strategy == 1:
            reward = multi_bandit.get_reward(selected_action)
        else:
            reward = multi_bandit.non_stationary_reward(selected_action)
        
        action_counts[selected_action] += 1
        est_values[selected_action] += (reward - est_values[selected_action]) / action_counts[selected_action]
        
        rewards_collected.append(reward)
        avg_rewards.append(avg_rewards[step - 1] + (reward - avg_rewards[step - 1]) / step)

    print("Action Counts:", action_counts)
    return est_values, avg_rewards, rewards_collected

# Non-stationary rewards
num_arms = 10
bandit_instance = MultiArmBandit(num_arms)
estimates, avg_rewards, rewards = epsilon_greedy_agent(bandit_instance, 0.2, 10000, strategy=2)
plt.plot(avg_rewards)
plt.ylabel("Average Reward")
plt.xlabel("Steps")

"""3. Modified epsilon-greedy agent to track non-stationary rewards."""

def modified_epsilon_greedy(bandit, eps, max_steps, strategy=1, alpha=0.7):
    estimated_values = [0] * bandit.num_arms
    rewards_recorded = []
    avg_rewards_over_time = [0]
    
    for step in range(1, max_steps):
        if random.random() > eps:
            selected_action = estimated_values.index(max(estimated_values))
        else:
            selected_action = random.choice(bandit.available_actions())
        
        if strategy == 1:
            reward = bandit.get_reward(selected_action)
        else:
            reward = bandit.non_stationary_reward(selected_action)
        
        estimated_values[selected_action] += (reward - estimated_values[selected_action]) * alpha
        rewards_recorded.append(reward)
        avg_rewards_over_time.append(avg_rewards_over_time[step - 1] + (reward - avg_rewards_over_time[step - 1]) / step)

    print("Estimated Values:", estimated_values)
    return estimated_values, avg_rewards_over_time, rewards_recorded

# Using the modified agent
estimates_mod, avg_rewards_mod, rewards_mod = modified_epsilon_greedy(bandit_instance, 0.2, 10000, strategy=2)
plt.plot(avg_rewards_mod)
plt.ylabel("Average Reward")
plt.xlabel("Steps")