# vortex_maze.py
import numpy as np
import random

class PythonMaze:
    def __init__(self, dim=11, seed=-1):
        self.dim = dim
        self.grid = np.ones((dim, dim), dtype=np.int32)
        
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            
        attempts = 0
        is_valid = False
        best_exit = None
        best_grid = None
        
        while not is_valid and attempts < 200:
            attempts += 1
            self.grid.fill(1)
            self.gen(1, 1)
            
            exit_params = self.find_hardest_exit()
            if best_exit is None or (exit_params['d'] + exit_params['turns'] > best_exit['d'] + best_exit['turns']):
                best_exit = exit_params
                best_grid = np.copy(self.grid)
            
            if exit_params['d'] >= 20 and exit_params['turns'] >= 5:
                is_valid = True
        
        self.grid = best_grid
        self.grid[best_exit['y']][best_exit['x']] = 2
        self.optimal_dist = best_exit['d']
        
    def gen(self, x, y):
        self.grid[y][x] = 0
        dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx * 2, y + dy * 2
            if 0 < nx < self.dim - 1 and 0 < ny < self.dim - 1:
                if self.grid[ny][nx] == 1:
                    self.grid[y + dy][x + dx] = 0
                    self.gen(nx, ny)
                    
    def find_hardest_exit(self):
        q = [{'x': 1, 'y': 1, 'd': 0, 'dx': 0, 'dy': 0, 'turns': 0}]
        visited = np.zeros((self.dim, self.dim), dtype=bool)
        visited[1][1] = True
        best = {'x': 1, 'y': 1, 'd': 0, 'turns': 0}
        max_score = 0
        
        while q:
            curr = q.pop(0)
            score = curr['d'] + curr['turns'] * 3
            if score > max_score and (curr['x'] != 1 or curr['y'] != 1):
                max_score = score
                best = curr
                
            for dx, dy in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                nx, ny = curr['x'] + dx, curr['y'] + dy
                if 0 < nx < self.dim - 1 and 0 < ny < self.dim - 1:
                    if not visited[ny][nx] and self.grid[ny][nx] == 0:
                        visited[ny][nx] = True
                        is_turn = (curr['dx'] != 0 or curr['dy'] != 0) and (curr['dx'] != dx or curr['dy'] != dy)
                        q.append({
                            'x': nx, 'y': ny, 'd': curr['d'] + 1,
                            'dx': dx, 'dy': dy, 'turns': curr['turns'] + (1 if is_turn else 0)
                        })
        return best
