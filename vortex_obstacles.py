# vortex_obstacles.py
import torch

def init_arena_obstacles(grid_x, grid_y, maze_grid, res, device):
    """
    Разделяет сетку лабиринта на динамические внутренние стены 
    и жесткие, монолитные внешние границы.
    """
    inner_obstacles = torch.zeros((1, 1, res, res), device=device)
    outer_obstacles = torch.zeros((1, 1, res, res), device=device)
    dim = maze_grid.shape[0]
    
    maze_tensor = torch.tensor(maze_grid, dtype=torch.float32, device=device)
    scale_limit = 0.8
    
    col_idx = ((grid_x + scale_limit) / (2.0 * scale_limit) * dim).long()
    row_idx = ((grid_y + scale_limit) / (2.0 * scale_limit) * dim).long()
    
    within_bounds = (torch.abs(grid_x) <= scale_limit) & (torch.abs(grid_y) <= scale_limit)
    
    valid_cols = col_idx[within_bounds].clamp(0, dim - 1)
    valid_rows = row_idx[within_bounds].clamp(0, dim - 1)
    
    # Выделяем крайние граничные ячейки сетки как внешние границы
    is_boundary = (valid_rows == 0) | (valid_rows == dim - 1) | (valid_cols == 0) | (valid_cols == dim - 1)
    
    # Любая точка вне scale_limit также считается жесткой внешней границей
    outer_obstacles[0, 0, ~within_bounds] = 1.0
    
    wall_vals = maze_tensor[valid_rows, valid_cols]
    
    # Разделяем препятствия на внутренние разрушаемые стены и жесткие внешние границы
    inner_obstacles[0, 0, within_bounds] = torch.where((wall_vals == 1.0) & (~is_boundary), 1.0, 0.0)
    outer_obstacles[0, 0, within_bounds] = torch.where((wall_vals == 1.0) & is_boundary, 1.0, 0.0)
    
    return inner_obstacles, outer_obstacles
