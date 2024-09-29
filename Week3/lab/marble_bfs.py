import heapq

class MarbleSolitaire:
    def __init__(self, board):
        self.board = board
        self.size = len(board)
        self.center = (self.size // 2, self.size // 2)

    def get_possible_moves(self):
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:
                    for di, dj in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            if self.board[ni][nj] == 0 and self.board[i + di // 2][j + dj // 2] == 1:
                                moves.append((i, j, ni, nj))
        return moves

    def make_move(self, move):
        from_x, from_y, to_x, to_y = move
        new_board = [row[:] for row in self.board]
        new_board[from_x][from_y] = 0
        new_board[to_x][to_y] = 1
        new_board[(from_x + to_x) // 2][(from_y + to_y) // 2] = 0
        return MarbleSolitaire(new_board)

    def is_goal(self):
        return sum(row.count(1) for row in self.board) == 1 and self.board[self.center[0]][self.center[1]] == 1

    def heuristic(self):
        """Heuristic: Number of marbles left."""
        return sum(row.count(1) for row in self.board)

    def __lt__(self, other):
        """Comparison operator for heapq."""
        return self.heuristic() < other.heuristic()

    def best_first_search(self):
        open_list = []
        heapq.heappush(open_list, (self.heuristic(), self, []))
        visited = set()
        visited.add(tuple(map(tuple, self.board)))

        while open_list:
            _, node, path = heapq.heappop(open_list)

            if node.is_goal():
                return path

            for move in node.get_possible_moves():
                child_node = node.make_move(move)
                board_tuple = tuple(map(tuple, child_node.board))

                if board_tuple not in visited:
                    new_path = path + [move]
                    heapq.heappush(open_list, (child_node.heuristic(), child_node, new_path))
                    visited.add(board_tuple)

        return None  # No solution found


def print_board(board):
    for row in board:
        print(" ".join(str(x) for x in row))
    print()


# Try a known solvable board configuration
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
solution = game.best_first_search()

if solution:
    print("Best-First Search solution found!")
    current_board = MarbleSolitaire(initial_board)
    print_board(current_board.board)

    for move in solution:
        current_board = current_board.make_move(move)
        print(f"Move: {move}")
        print_board(current_board.board)
else:
    print("No solution found.")
