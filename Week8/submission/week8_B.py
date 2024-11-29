import numpy as np

# Parameters for the problem
gamma = 0.9  # Discount factor
transfer_cost = 2  # Cost of moving bikes
profit_per_rental = 10  # Reward for each rental
max_inventory = 20  # Maximum number of bikes at each location
max_transfer = 5  # Maximum bikes to move between locations
expected_requests_loc1 = 3  # Expected rental requests at location 1
expected_requests_loc2 = 4  # Expected rental requests at location 2
expected_returns_loc1 = 3  # Expected bike returns at location 1
expected_returns_loc2 = 2  # Expected bike returns at location 2

# Initialize value function and policy
state_values = np.zeros((max_inventory + 1, max_inventory + 1))
policy_table = np.zeros((max_inventory + 1, max_inventory + 1), dtype=int)

# Define action space
possible_transfers = np.arange(-max_transfer, max_transfer + 1)

# Define Poisson probability function
def poisson_prob(n, lam):
    return (lam**n) * np.exp(-lam) / np.math.factorial(n)

# Evaluate Policy
def policy_evaluation():
    delta_threshold = 1e-6
    while True:
        max_delta = 0
        for bikes_loc1 in range(max_inventory + 1):
            for bikes_loc2 in range(max_inventory + 1):
                previous_value = state_values[bikes_loc1, bikes_loc2]
                current_action = policy_table[bikes_loc1, bikes_loc2]
                expected_reward = 0
                expected_value = 0

                for rental_loc1 in range(expected_requests_loc1 + 1):
                    for rental_loc2 in range(expected_requests_loc2 + 1):
                        for return_loc1 in range(expected_returns_loc1 + 1):
                            for return_loc2 in range(expected_returns_loc2 + 1):
                                actual_rentals_loc1 = min(rental_loc1, bikes_loc1)
                                actual_rentals_loc2 = min(rental_loc2, bikes_loc2)

                                rental_revenue_loc1 = actual_rentals_loc1 * profit_per_rental
                                rental_revenue_loc2 = actual_rentals_loc2 * profit_per_rental

                                prob_rent_loc1 = poisson_prob(rental_loc1, expected_requests_loc1)
                                prob_rent_loc2 = poisson_prob(rental_loc2, expected_requests_loc2)
                                prob_return_loc1 = poisson_prob(return_loc1, expected_returns_loc1)
                                prob_return_loc2 = poisson_prob(return_loc2, expected_returns_loc2)

                                total_prob = prob_rent_loc1 * prob_rent_loc2 * prob_return_loc1 * prob_return_loc2

                                next_loc1 = min(bikes_loc1 - actual_rentals_loc1 + return_loc1, max_inventory)
                                next_loc2 = min(bikes_loc2 - actual_rentals_loc2 + return_loc2, max_inventory)

                                expected_value += total_prob * state_values[next_loc1, next_loc2]
                                expected_reward += total_prob * (rental_revenue_loc1 + rental_revenue_loc2)

                expected_reward -= transfer_cost * abs(current_action)
                state_values[bikes_loc1, bikes_loc2] = expected_reward + gamma * expected_value
                max_delta = max(max_delta, abs(previous_value - state_values[bikes_loc1, bikes_loc2]))
        
        if max_delta < delta_threshold:
            break

# Improve Policy
def policy_improvement():
    policy_stable = True
    for bikes_loc1 in range(max_inventory + 1):
        for bikes_loc2 in range(max_inventory + 1):
            old_action = policy_table[bikes_loc1, bikes_loc2]
            action_values = np.zeros(possible_transfers.shape)

            for idx, action in np.ndenumerate(possible_transfers):
                if 0 <= bikes_loc1 - action <= max_inventory and 0 <= bikes_loc2 + action <= max_inventory:
                    action_values[idx] = -abs(action) * transfer_cost
                    for rental_loc1 in range(expected_requests_loc1 + 1):
                        for rental_loc2 in range(expected_requests_loc2 + 1):
                            for return_loc1 in range(expected_returns_loc1 + 1):
                                for return_loc2 in range(expected_returns_loc2 + 1):
                                    rentals_loc1 = min(rental_loc1, bikes_loc1 - action)
                                    rentals_loc2 = min(rental_loc2, bikes_loc2 + action)

                                    reward_rentals = (rentals_loc1 + rentals_loc2) * profit_per_rental

                                    final_loc1 = min(bikes_loc1 - action - rentals_loc1 + return_loc1, max_inventory)
                                    final_loc2 = min(bikes_loc2 + action - rentals_loc2 + return_loc2, max_inventory)

                                    prob = (
                                        poisson_prob(rental_loc1, expected_requests_loc1)
                                        * poisson_prob(rental_loc2, expected_requests_loc2)
                                        * poisson_prob(return_loc1, expected_returns_loc1)
                                        * poisson_prob(return_loc2, expected_returns_loc2)
                                    )
                                    action_values[idx] += prob * (reward_rentals + gamma * state_values[final_loc1, final_loc2])
            
            best_action = possible_transfers[np.argmax(action_values)]
            policy_table[bikes_loc1, bikes_loc2] = best_action

            if old_action != best_action:
                policy_stable = False

    return policy_stable

# Perform Policy Iteration
while True:
    policy_evaluation()
    if policy_improvement():
        break

# Results
print("Optimal Policy:")
print(policy_table)
print("State Values:")
print(state_values)