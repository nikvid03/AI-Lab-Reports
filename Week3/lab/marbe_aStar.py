import heapq

class MarbleSolitaire:
    def __init__(self, board):
        self.board = board
        self.size = len(board)
        self.center = (self.size // 2, self.size // 2)  # Center position of the board

    def get_possible_moves(self):
        """Find all valid moves (jump over a marble) for the current board state."""
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:  # If there's a marble
                    # Check all four possible jump directions (up, down, left, right)
                    for di, dj in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            if self.board[ni][nj] == 0 and self.board[i + di // 2][j + dj // 2] == 1:
                                # Valid move (from i,j to ni,nj, jumping over the marble in between)
                                moves.append((i, j, ni, nj))
        return moves

    def make_move(self, move):
        """Perform the move and return a new board state."""
        from_x, from_y, to_x, to_y = move
        new_board = [row[:] for row in self.board]
        new_board[from_x][from_y] = 0
        new_board[to_x][to_y] = 1
        new_board[(from_x + to_x) // 2][(from_y + to_y) // 2] = 0  # Remove jumped-over marble
        return MarbleSolitaire(new_board)

    def is_goal(self):
        """Check if the goal state is reached: one marble left at the center."""
        return sum(row.count(1) for row in self.board) == 1 and self.board[self.center[0]][self.center[1]] == 1

    def heuristic(self):
        """Heuristic: number of marbles left."""
        return sum(row.count(1) for row in self.board)

    def __lt__(self, other):
        """Comparison operator for heapq to avoid errors when heuristic values are equal."""
        return True  # Arbitrarily return True since they can't really be compared directly

    def a_star_search(self):
        """Perform A* search to solve the Marble Solitaire puzzle."""
        open_list = []
        heapq.heappush(open_list, (self.heuristic(), self, []))
        visited = set()

        while open_list:
            _, node, path = heapq.heappop(open_list)

            if tuple(map(tuple, node.board)) in visited:
                continue

            visited.add(tuple(map(tuple, node.board)))

            if node.is_goal():
                return path

            for move in node.get_possible_moves():
                child_node = node.make_move(move)
                new_path = path + [move]
                heapq.heappush(open_list, (len(new_path) + child_node.heuristic(), child_node, new_path))

        return None  # No solution found


def print_board(board):
    for row in board:
        print(" ".join(str(x) for x in row))
    print()


# Example initial board setup (7x7 cross pattern)
initial_board = [
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0]
]

game = MarbleSolitaire(initial_board)
solution = game.a_star_search()

if solution:
    print("Solution found!")
    current_board = MarbleSolitaire(initial_board)
    print_board(current_board.board)

    for move in solution:
        current_board = current_board.make_move(move)
        print(f"Move: {move}")
        print_board(current_board.board)
else:
    print("No solution found.")
