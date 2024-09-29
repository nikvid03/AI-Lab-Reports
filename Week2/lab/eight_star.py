import heapq
import random

# Author : Pratik Shah
# Date : Sept 4, 2024
# Place : IIIT Vadodara

# Course : CS307 Artificial Intelligence
# Exercise : Puzzle Eight Solver
# Learning Objective : Revisit concepts of basic data structures, BFS and DFS

class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g  # distance to root
        self.h = h  # estimated distance to goal
        self.f = g + h  # evaluation function

    def __lt__(self, other):
        return self.f < other.f  # Use f for comparison

def heuristic(state, goal_state):
    # Heuristic: Count the number of misplaced tiles
    h = sum(1 for i in range(len(state)) if state[i] != goal_state[i] and state[i] != 0)
    return h

def get_successors(node):
    successors = []
    index = node.state.index(0)  # Find the index of the empty space (0)
    moves = []
    # Determine possible moves based on current index
    if index // 3 > 0:  # Can move up
        moves.append(-3)
    if index // 3 < 2:  # Can move down
        moves.append(3)
    if index % 3 > 0:  # Can move left
        moves.append(-1)
    if index % 3 < 2:  # Can move right
        moves.append(1)

    for move in moves:
        new_index = index + move
        new_state = list(node.state)
        # Swap the empty space with the adjacent tile
        new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
        h = heuristic(new_state, [1, 2, 3, 4, 5, 6, 7, 8, 0])  # Calculate the heuristic for the new state
        successor = Node(new_state, node, node.g + 1, h)
        successors.append(successor)            

    return successors

def search_agent(start_state, goal_state):
    start_node = Node(start_state, None, 0, heuristic(start_state, goal_state))
    frontier = []
    heapq.heappush(frontier, start_node)
    visited = set()
    
    while frontier:
        node = heapq.heappop(frontier)
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))
        
        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1]  # Return the path in the correct order
        
        for successor in get_successors(node):
            heapq.heappush(frontier, successor)

    return None

# Initialize the start state and generate a random goal state
start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]  # Set a valid goal state for testing

# Run the search algorithm
solution = search_agent(start_state, goal_state)
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
