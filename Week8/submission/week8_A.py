import numpy as np
import random

'''=============================================
Initialization
============================================='''

# Hyperparameters
THRESHOLD = 0.05
DISCOUNT = 0.99
RANDOMNESS = 0.1  

# Define all states in the grid
state_space = []
for row in range(3):
    for col in range(4):
        state_space.append((row, col))

# Assign rewards to states
state_rewards = {}
for state in state_space:
    if state == (1, 3):
        state_rewards[state] = -1
    elif state == (2, 3):
        state_rewards[state] = 1
    else:
        state_rewards[state] = -0.04

# Define actions for each state (excluding terminal states)
action_space = {
    (0, 0): ('D', 'R'), 
    (0, 1): ('D', 'R', 'L'),    
    (0, 2): ('D', 'L', 'R'),
    (0, 3): ('D', 'L'),
    (1, 2): ('U', 'D', 'L', 'R'),
    (1, 0): ('D', 'U', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('U', 'L', 'R'),
    (2, 2): ('U', 'L', 'R'),
}

# Initialize policy randomly
policy_map = {}
for state in action_space.keys():
    policy_map[state] = np.random.choice(action_space[state])

# Initialize the value function
value_function = {}
for state in state_space:
    if state in action_space.keys():
        value_function[state] = 0
    if state == (1, 3):
        value_function[state] = -1
    if state == (2, 3):
        value_function[state] = 1
    if state == (1, 1):
        value_function[state] = 0

'''=============================================
Value Iteration
============================================='''

iteration_count = 0
while True:
    max_change = 0
    for current_state in state_space:
        if current_state in policy_map:
            
            old_value = value_function[current_state]
            best_value = -1e6
            
            for action in action_space[current_state]:
                if action == 'U':
                    next_state = [current_state[0] - 1, current_state[1]]
                if action == 'D':
                    next_state = [current_state[0] + 1, current_state[1]]
                if action == 'L':
                    next_state = [current_state[0], current_state[1] - 1]
                if action == 'R':
                    next_state = [current_state[0], current_state[1] + 1]

                # Introduce random action transitions
                alternative_action = np.random.choice([act for act in action_space[current_state] if act != action])
                random_number = random.randint(0, 100)
                if 80 < random_number <= 90:
                    if alternative_action == 'U':
                        alternative_action = 'L'
                    if alternative_action == 'D':
                        alternative_action = 'R'
                    if alternative_action == 'L':
                        alternative_action = 'D'
                    if alternative_action == 'R':
                        alternative_action = 'U'
                if random_number > 90:
                    if alternative_action == 'U':
                        alternative_action = 'R'
                    if alternative_action == 'D':
                        alternative_action = 'L'
                    if alternative_action == 'L':
                        alternative_action = 'U'
                    if alternative_action == 'R':
                        alternative_action = 'D'
                if alternative_action == 'U':
                    random_state = [current_state[0] - 1, current_state[1]]
                if alternative_action == 'D':
                    random_state = [current_state[0] + 1, current_state[1]]
                if alternative_action == 'L':
                    random_state = [current_state[0], current_state[1] - 1]
                if alternative_action == 'R':
                    random_state = [current_state[0], current_state[1] + 1]

                if (0 <= random_state[0] < 3) and (0 <= random_state[1] < 4): 
                    next_state_tuple = tuple(next_state)
                    random_state_tuple = tuple(random_state)
                    calc_value = state_rewards[current_state] + DISCOUNT * (
                        (1 - RANDOMNESS) * value_function[next_state_tuple] +
                        RANDOMNESS * value_function[random_state_tuple]
                    )
                    if calc_value > best_value: 
                        best_value = calc_value
                        policy_map[current_state] = action
                                
            value_function[current_state] = best_value
            max_change = max(max_change, abs(old_value - value_function[current_state]))
         
    # Stop if change is below the threshold
    if max_change < THRESHOLD:
        break
    iteration_count += 1

# Print results
print(f"Total iterations: {iteration_count}")

for state, value in value_function.items():
    print(f"State: {state}, Value: {value}")