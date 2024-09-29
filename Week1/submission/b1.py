from collections import deque


# Check if the current state is the goal state
def is_goal(state):
    return state == [1, 1, 1, 0, -1, -1, -1]


# Generate all possible successor states
def get_successors(state):
    successors = []
    empty_index = state.index(0)  # 0 represents the empty space
    moves = [-1, -2, 1, 2]  # Possible moves: swap with adjacent or 2 spaces away

    for move in moves:
        new_index = empty_index + move
        if 0 <= new_index < len(state):  # Ensure new index is within bounds
            new_state = state[:]
            new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
            successors.append(new_state)

    return successors


# Breadth-first search to find the solution
def bfs(start_state):
    queue = deque([(start_state, [])])
    visited = set()
    while queue:
        current_state, path = queue.popleft()
        if tuple(current_state) in visited:
            continue
        visited.add(tuple(current_state))
        path = path + [current_state]

        if is_goal(current_state):
            return path

        for successor in get_successors(current_state):
            queue.append((successor, path))

    return None


# Initialize the start state
start_state = [-1, -1, -1, 0, 1, 1, 1]

# Run the BFS algorithm to find the solution
solution = bfs(start_state)


if solution:
    print("Solution found with", len(solution) - 1, "steps:")
    for step in solution:
        print(step)
else:
    print("No solution found.")