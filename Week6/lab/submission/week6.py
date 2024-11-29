import numpy as np

class AssociativeMemory:
    def _init_(self, dimension):
        self.dimension = dimension
        self.connection_matrix = np.zeros((dimension, dimension))

    def learn_patterns(self, input_patterns):
        for pattern in input_patterns:
            self.connection_matrix += np.outer(pattern, pattern)
        np.fill_diagonal(self.connection_matrix, 0)  # Avoid self-loops

    def retrieve_pattern(self, input_pattern, iterations=10):
        for _ in range(iterations):
            input_pattern = np.sign(self.connection_matrix @ input_pattern)
        return input_pattern


# Example demonstration:
grid_size = 100  # Represents a 10x10 grid
memory = AssociativeMemory(grid_size)

# Generate random patterns with binary values
training_patterns = [np.random.choice([-1, 1], grid_size) for _ in range(3)]
memory.learn_patterns(training_patterns)

# Test the retrieval process with altered input
altered_input = training_patterns[0] + np.random.choice([-1, 0, 1], grid_size)
altered_input = np.sign(altered_input)
retrieved_pattern = memory.retrieve_pattern(altered_input)

print("Retrieved Pattern:", retrieved_pattern)