import random
import math
import matplotlib.pyplot as plt

def calculate_distance(loc1, loc2):
    x1, y1 = loc1
    x2, y2 = loc2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_tour_distance(tour, locations):
    total_distance = 0
    for i in range(len(tour) - 1):
        node1 = tour[i]
        node2 = tour[i + 1]
        if node1 not in locations or node2 not in locations:
            return float('inf')  # Return infinity distance if node not found
        total_distance += calculate_distance(locations[node1], locations[node2])
    # Return to the starting point
    total_distance += calculate_distance(locations[tour[-1]], locations[tour[0]])
    return total_distance

def simulated_annealing(locations, initial_temperature, cooling_rate, iterations):
    num_locations = len(locations)
    current_tour = list(locations.keys())  # Initial tour order
    random.shuffle(current_tour)  # Random initial tour
    current_distance = calculate_tour_distance(current_tour, locations)

    best_tour = current_tour.copy()
    best_distance = current_distance

    temperature = initial_temperature

    distances = []  # Store best distances for each iteration

    for iteration in range(iterations):
        # Generate a neighboring solution by swapping two random cities
        neighbor_tour = current_tour.copy()
        i, j = random.sample(range(num_locations), 2)
        neighbor_tour[i], neighbor_tour[j] = neighbor_tour[j], neighbor_tour[i]
        neighbor_distance = calculate_tour_distance(neighbor_tour, locations)

        # Acceptance probability
        if neighbor_distance < current_distance or random.random() < acceptance_probability(current_distance, neighbor_distance, temperature):
            current_tour = neighbor_tour
            current_distance = neighbor_distance

        # Update the best solution if necessary
        if current_distance < best_distance:
            best_tour = current_tour.copy()
            best_distance = current_distance

        # Cooling
        temperature *= cooling_rate
        
        # Store best distance for this iteration
        distances.append(best_distance)

    return best_tour, best_distance, distances

def acceptance_probability(current_distance, neighbor_distance, temperature):
    if neighbor_distance < current_distance:
        return 1.0
    else:
        return math.exp((current_distance - neighbor_distance) / temperature)

def read_tsp_file(file_path):
    locations = {}
    with open(file_path, 'r') as file:
        found_section = False
        for line in file:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                found_section = True
                continue
            if found_section and line == "EOF":
                break
            if found_section:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        locations[node_id] = (x, y)
                    except ValueError:
                        print("Error parsing line:", line)
                else:
                    print("Invalid line format:", line)
    return locations

if _name_ == "_main_":
    # Read .tsp file and extract locations
    # file_path = "xqf131.tsp"  # Replace with your file path
    locations = read_tsp_file('xqf131.tsp')

    if not locations:
        print("No locations found.")
    else:
        # Simulated Annealing parameters
        initial_temperature = 10000  # Increase initial temperature
        cooling_rate = 0.999  # Slower cooling rate
        iterations_values = [10000 * i for i in range(1, 21)]  # iterations values 10000, 20000, ..., up to 20 values

        for iterations in iterations_values:
            # Run simulated annealing
            best_tour, best_distance, distances = simulated_annealing(locations, initial_temperature, cooling_rate, iterations)

            # Print best tour and distance
            print(f"Iterations: {iterations}, Best Tour Distance: {best_distance}")

            # Plot graph
            plt.plot(range(iterations), distances, label=f"Iterations: {iterations}")

        plt.xlabel("Iterations")
        plt.ylabel("Best Tour Distance")
        plt.legend()
        plt.show()