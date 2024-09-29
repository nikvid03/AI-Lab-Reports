import numpy as np
import random
import matplotlib.pyplot as plt

# Energy function to compute edge differences between adjacent tiles
def compute_energy(tiles):
    energy = 0
    # Compare each tile with its right and bottom neighbors (if any)
    for i in range(grid_size):
        for j in range(grid_size):
            if j < grid_size - 1:
                energy += np.sum(np.abs(tiles[i, j][:, -1] - tiles[i, j + 1][:, 0]))  # Right edge comparison
            if i < grid_size - 1:
                energy += np.sum(np.abs(tiles[i, j][-1, :] - tiles[i + 1, j][0, :]))  # Bottom edge comparison
    return energy

# Simulated annealing function
def simulated_annealing(tiles, max_iter=10000, initial_temp=1.0, cooling_rate=0.99):
    current_state = tiles.copy()
    current_energy = compute_energy(current_state)
    temperature = initial_temp
    
    for iteration in range(max_iter):
        # Swap two random tiles
        i1, j1 = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        i2, j2 = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        new_state = current_state.copy()
        new_state[i1, j1], new_state[i2, j2] = new_state[i2, j2], new_state[i1, j1]
        
        # Calculate energy of the new state
        new_energy = compute_energy(new_state)
        energy_diff = new_energy - current_energy
        
        # Decide whether to accept the new state
        if energy_diff < 0 or random.random() < np.exp(-energy_diff / temperature):
            current_state = new_state
            current_energy = new_energy
        
        # Cool down the system
        temperature *= cooling_rate
        
        # Optionally, print the progress or display the current puzzle arrangement
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Energy: {current_energy}")
            plt.imshow(np.block(current_state), cmap='gray')
            plt.show()

    return current_state

# Assuming 'tiles' is a 2D array where each entry is a tile from the scrambled image
final_state = simulated_annealing(tiles)
