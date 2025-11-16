import numpy as np

class GridWorld:
    def __init__(self, n=4, gamma=1.0):
        self.n = n
        self.gamma = gamma
        self.states = list(range(n*n))
        self.terminal_states = [0, n*n - 1]  
        self.actions = [0, 1, 2, 3]  

    def step(self, state, action):
       
        if state in self.terminal_states:
            return state, 0

        row, col = divmod(state, self.n)

        if action == 0:    # UP
            next_row, next_col = max(row - 1, 0), col
        elif action == 1:  # RIGHT
            next_row, next_col = row, min(col + 1, self.n - 1)
        elif action == 2:  # DOWN
            next_row, next_col = min(row + 1, self.n - 1), col
        else:              # LEFT
            next_row, next_col = row, max(col - 1, 0)

        next_state = next_row * self.n + next_col

        if next_state == state:
            return state, -1
        
        return next_state, -1