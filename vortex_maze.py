# vortex_maze.py
import numpy as np
import random

class PythonMaze:
    def __init__(self, dim=13, seed=-1):
        self.dim = dim
        self.grid = np.ones((dim, dim), dtype=np.int32)
        
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            
        self.grid.fill(1)
        self.gen(1, 1)
        
        # Очищаем все старые маркеры портала (двойки), нам они больше не нужны
        self.grid[self.grid == 2] = 0
        
        # Создаем Алхимический Котел (вырубаем площадку 3x3 ровно в центре)
        self.cx, self.cy = dim // 2, dim // 2
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if 0 < self.cy + dy < dim-1 and 0 < self.cx + dx < dim-1:
                    self.grid[self.cy + dy][self.cx + dx] = 0
                
        self.cauldron_cell = (self.cx, self.cy)
        
        # Расчищаем 4 угла лабиринта, чтобы никто не застрял при спавне
        self.grid[1][1] = 0
        self.grid[dim-2][1] = 0
        self.grid[1][dim-2] = 0
        self.grid[dim-2][dim-2] = 0
        
        # Сущности будут жить в трех углах
        self.entity_cells = [(1, 1), (dim-2, 1), (1, dim-2)]
        
        # Юзер появляется в четвертом углу
        self.spawn_cell = (dim-2, dim-2)

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
