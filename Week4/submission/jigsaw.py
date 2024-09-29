import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the scrambled image from the .mat file
data = loadmat('scrambled_lena.mat')
image_data = data['scrambled_lena']  # Ensure this key matches the .mat structure

# Parameters
TILE_COUNT = 9  # Number of pieces (3x3 puzzle)
START_TEMP = 1000  # Initial temperature for simulated annealing
TEMP_DECAY = 0.99  # Cooling rate
ITERATION_LIMIT = 10000  # Number of iterations for annealing

# Function to show an image with an optional title
def show_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to calculate the dissimilarity score for puzzle configuration
def evaluate_puzzle_fitness(arrangement, pieces):
    fitness_score = 0
    grid_size = int(np.sqrt(len(pieces)))

    for row in range(grid_size):
        for col in range(grid_size):
            current_tile = pieces[arrangement[row * grid_size + col]]
            # Compare with tile above (top-bottom match)
            if row > 0:
                above_tile = pieces[arrangement[(row-1) * grid_size + col]]
                fitness_score += np.sum(np.abs(current_tile[0, :] - above_tile[-1, :]))
            # Compare with tile to the left (left-right match)
            if col > 0:
                left_tile = pieces[arrangement[row * grid_size + (col-1)]]
                fitness_score += np.sum(np.abs(current_tile[:, 0] - left_tile[:, -1]))
    return fitness_score

# Function to shuffle two randomly selected tiles
def swap_random_tiles(arrangement):
    new_arrangement = arrangement.copy()
    tile_a, tile_b = random.sample(range(len(arrangement)), 2)
    new_arrangement[tile_a], new_arrangement[tile_b] = new_arrangement[tile_b], new_arrangement[tile_a]
    return new_arrangement

# Simulated annealing to find the optimal arrangement of puzzle pieces
def optimize_puzzle_arrangement(pieces):
    current_state = list(range(len(pieces)))  # Initial arrangement of tiles
    current_fitness = evaluate_puzzle_fitness(current_state, pieces)
    temperature = START_TEMP

    for step in range(ITERATION_LIMIT):
        # Get a new state by swapping two pieces
        new_state = swap_random_tiles(current_state)
        new_fitness = evaluate_puzzle_fitness(new_state, pieces)

        # Accept new arrangement if it's better or probabilistically based on temperature
        if new_fitness < current_fitness or random.random() < np.exp((current_fitness - new_fitness) / temperature):
            current_state = new_state
            current_fitness = new_fitness

        # Cool down the temperature
        temperature *= TEMP_DECAY

        # Periodically log the progress
        if step % 1000 == 0:
            print(f"Step {step}, Fitness: {current_fitness}, Temperature: {temperature}")

        # Terminate early if puzzle is solved (fitness score is 0)
        if current_fitness == 0:
            break

    return current_state

# Function to break the image into smaller square tiles
def cut_image_into_tiles(image, num_tiles):
    height, width = image.shape
    tile_height, tile_width = height // num_tiles, width // num_tiles
    return [image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]
            for i in range(num_tiles) for j in range(num_tiles)]

# Function to reconstruct the image from the tile arrangement
def reconstruct_image(arrangement, pieces, num_tiles):
    tile_height, tile_width = pieces[0].shape
    reconstructed_image = np.zeros((tile_height * num_tiles, tile_width * num_tiles), dtype=pieces[0].dtype)

    for row in range(num_tiles):
        for col in range(num_tiles):
            reconstructed_image[row * tile_height:(row + 1) * tile_height, col * tile_width:(col + 1) * tile_width] = pieces[arrangement[row * num_tiles + col]]

    return reconstructed_image

# Main execution
if __name__ == "__main__":
    # Display the scrambled puzzle image
    show_image(image_data, "Scrambled Puzzle")

    # Split the scrambled image into individual tiles
    puzzle_tiles = cut_image_into_tiles(image_data, int(np.sqrt(TILE_COUNT)))

    # Use simulated annealing to solve the puzzle
    final_tile_order = optimize_puzzle_arrangement(puzzle_tiles)

    # Rebuild and display the solved image
    solved_image = reconstruct_image(final_tile_order, puzzle_tiles, int(np.sqrt(TILE_COUNT)))
    show_image(solved_image, "Solved Puzzle")
