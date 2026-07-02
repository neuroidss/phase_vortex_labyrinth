# vortex_physics.py
import torch
import torch.nn.functional as F
import math
import numpy as np
from vortex_fluid import FluidSolver
from vortex_maze import PythonMaze

try:
    from implicit_config import COORDS_16_X, COORDS_16_Y
except ImportError:
    COORDS_16_X = [10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14]
    COORDS_16_Y = [-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71]

class PhaseVortexArena:
    def __init__(self, device, width, height, res):
        self.device = device
        self.WIDTH, self.HEIGHT = width, height
        self.res = res
        self.solver = FluidSolver(res, device)
        
        self.cfg = {'vorticity_sensitivity': 0.15, 'fluid_damping': 0.80}
        self.smooth_vorticity = 0.0
        
        self.u = torch.zeros((1, 1, res, res), device=device)
        self.v = torch.zeros((1, 1, res, res), device=device)
        self.density = torch.zeros((1, 3, res, res), device=device)
        self.player_density = torch.zeros((1, 1, res, res), device=device)
        self.wall_density = torch.zeros((1, 1, res, res), device=device)
        
        self.pin_x = torch.tensor(COORDS_16_X, dtype=torch.float32, device=device)
        self.pin_y = torch.tensor(COORDS_16_Y, dtype=torch.float32, device=device)
        
        # Динамические частицы (когерентности), образующие тело слайма
        self.pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
        self.pin_captured = torch.zeros(16, dtype=torch.bool, device=device)
        
        self.player_pos = torch.tensor([width/2, height/2], dtype=torch.float32, device=device)
        self.portal_pos = torch.zeros(2, dtype=torch.float32, device=device)
        self.player_angle = 0.0
        self.screen_size = torch.tensor([width, height], dtype=torch.float32, device=device)
        
        self.y_indices, self.x_indices = torch.meshgrid(
            torch.arange(res, device=device, dtype=torch.float32),
            torch.arange(res, device=device, dtype=torch.float32), indexing='ij'
        )
        self.reset_world()
        
    def init_obstacles(self):
        # Вокруг лабиринта - ПУСТОТА (0.0)
        obstacles = torch.zeros((1, 1, self.res, self.res), device=self.device)
        dim = self.maze.dim
        goal_rows, goal_cols = np.where(self.maze.grid == 2)
        self.goal_cell = (goal_cols[0], goal_rows[0]) if len(goal_rows) > 0 else (dim - 2, dim - 2)
            
        maze_tensor = torch.tensor(self.maze.grid, dtype=torch.float32, device=self.device)
        scale_limit = 0.8
        
        col_idx = ((self.solver.grid_x + scale_limit) / (2.0 * scale_limit) * dim).long()
        row_idx = ((self.solver.grid_y + scale_limit) / (2.0 * scale_limit) * dim).long()
        
        within_bounds = (torch.abs(self.solver.grid_x) <= scale_limit) & (torch.abs(self.solver.grid_y) <= scale_limit)
        
        # Стены появляются ТОЛЬКО в границах матрицы лабиринта
        valid_cols = col_idx[within_bounds].clamp(0, dim - 1)
        valid_rows = row_idx[within_bounds].clamp(0, dim - 1)
        obstacles[0, 0, within_bounds] = torch.where(maze_tensor[valid_rows, valid_cols] == 1.0, 1.0, 0.0)
        
        return obstacles

    def reset_world(self):
        self.maze = PythonMaze(11)
        self.orig_obstacles = self.init_obstacles()
        
        self.cell_w = (self.WIDTH * 0.8) / self.maze.dim
        spawn_pos = (self.WIDTH * 0.1) + 1.5 * self.cell_w
        
        self.portal_pos[0] = (self.WIDTH * 0.1) + (self.goal_cell[0] + 0.5) * self.cell_w
        self.portal_pos[1] = (self.HEIGHT * 0.1) + (self.goal_cell[1] + 0.5) * self.cell_w
        
        self.player_pos.copy_(torch.tensor([spawn_pos, spawn_pos], dtype=torch.float32, device=self.device))
        self.player_angle = 0.0
        
        self.pin_captured.zero_()
        # Спавним слайма КОМПАКТНО, чтобы он точно помещался в центр клетки
        self.pin_pos[:, 0] = spawn_pos + self.pin_x * 1.5
        self.pin_pos[:, 1] = spawn_pos + self.pin_y * 1.5
        
        self.u.zero_()
        self.v.zero_()
        self.density.zero_()
        self.player_density.zero_()
        self.wall_density.copy_(self.orig_obstacles)
        self.smooth_vorticity = 0.0

    def step(self, dt, time_sec, eeg_c0, eeg_vx, eeg_vy, eeg_tq, is_real_data, compression, scale_factor):
        # Масштаб слайма подогнан так, чтобы в разжатом виде он не упирался в стены (max 2.2)
        scale = 0.2 + (1.0 - compression) * 2.0
        node_radius = 4.0 + (1.0 - compression) * 10.0

        self.u = torch.nan_to_num(self.u, nan=0.0) * 0.99
        self.v = torch.nan_to_num(self.v, nan=0.0) * 0.99
        self.density = torch.nan_to_num(self.density, nan=0.0) * 0.985
        self.u = torch.clamp(self.u, -250.0, 250.0)
        self.v = torch.clamp(self.v, -250.0, 250.0)
        
        self.wall_density = self.solver.advect(self.wall_density, self.u, self.v, dt * 0.08)
        erosion = torch.sqrt(self.u**2 + self.v**2) * 0.04 * dt
        self.wall_density = torch.clamp(self.wall_density - erosion, 0.0, 1.0)
        self.wall_density += (self.orig_obstacles - self.wall_density) * 0.35 * dt
        self.wall_density = torch.clamp(self.wall_density, 0.0, 1.0)

        # 1. ОБНОВЛЕНИЕ ДИНАМИЧЕСКИХ ЧАСТИЦ (Мягкое тело слайма)
        if self.pin_captured.all():
            self.reset_world()
            return
            
        # ЛОР-ФИКС: Общий центр масс считается по ВСЕМ точкам (включая захваченные в портале!),
        # это создает физический эффект последовательного затягивания "жгута"
        com = self.pin_pos.mean(dim=0)
        
        pin_uv = (self.pin_pos / self.screen_size) * 2.0 - 1.0
        sampled_u = F.grid_sample(self.u, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
        sampled_v = F.grid_sample(self.v, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
        
        fluid_vel = torch.stack([sampled_u, sampled_v], dim=1) * 80.0 
        
        # Упругость слайма
        cos_p, sin_p = math.cos(self.player_angle), math.sin(self.player_angle)
        dx_ideal = self.pin_x * cos_p - self.pin_y * sin_p
        dy_ideal = self.pin_x * sin_p + self.pin_y * cos_p
        ideal_pos = com.unsqueeze(0) + torch.stack([dx_ideal, dy_ideal], dim=1) * scale
        
        # Сила упругости
        f_spring = (ideal_pos - self.pin_pos) * 6.0
        
        # ВПРЫСКИВАЕМ упругость формы как импульс силы внутрь самой сетки жидкости!
        # Координаты точек пинов больше не двигаются этой силой напрямую, пролетая сквозь стены.
        # Жидкость течет к идеальной форме, а маркеры слайма плывут по этому течению, упираясь в стены.
        pin_gx = (self.pin_pos[:, 0] / self.WIDTH) * self.res
        pin_gy = (self.pin_pos[:, 1] / self.HEIGHT) * self.res
        dx_shape = self.x_indices.unsqueeze(0) - pin_gx.view(16, 1, 1)
        dy_shape = self.y_indices.unsqueeze(0) - pin_gy.view(16, 1, 1)
        node_footprint = torch.exp(-(dx_shape**2 + dy_shape**2) / 6.0) * (~self.pin_captured).float().view(16, 1, 1)
        
        spring_force_grid_x = torch.sum(node_footprint * f_spring[:, 0].view(16, 1, 1), dim=0)
        spring_force_grid_y = torch.sum(node_footprint * f_spring[:, 1].view(16, 1, 1), dim=0)
        self.u[0, 0] += spring_force_grid_x * dt
        self.v[0, 0] += spring_force_grid_y * dt
        
        # Отталкивание от стен (градиент плотности)
        w_val = F.grid_sample(self.wall_density, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_pad = F.pad(self.wall_density, (1, 1, 1, 1), mode='replicate')
        grad_x = 0.5 * (w_pad[:, :, 1:-1, 2:] - w_pad[:, :, 1:-1, :-2])
        grad_y = 0.5 * (w_pad[:, :, 2:, 1:-1] - w_pad[:, :, :-2, 1:-1])
        w_gx = F.grid_sample(grad_x, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_gy = F.grid_sample(grad_y, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        
        grad_norm = torch.sqrt(w_gx**2 + w_gy**2) + 1e-5
        dir_out_x = -w_gx / grad_norm
        dir_out_y = -w_gy / grad_norm
        f_wall = torch.stack([dir_out_x * w_val * 400.0, dir_out_y * w_val * 400.0], dim=1)
        
        # Захват когерентностей порталом при касании (0.35 от размера клетки - строго контакт!)
        dist_to_portal = torch.norm(self.pin_pos - self.portal_pos.unsqueeze(0), dim=1)
        capture_radius = self.cell_w * 0.35
        self.pin_captured = self.pin_captured | (dist_to_portal < capture_radius)
        cap_mask_f = self.pin_captured.float().unsqueeze(1)
        
        # ИТОГОВОЕ ДВИЖЕНИЕ: маркеры слайма плывут строго за жидкостью + локальный отскок от стен
        pin_vel = fluid_vel * 0.9 + f_wall
        
        # === СТРОГИЙ ЛИМИТ СКОРОСТИ ЧАСТИЦ (Защита от пролета сквозь стены) ===
        pin_vel = torch.clamp(pin_vel, -120.0, 120.0)
        
        self.pin_pos[~self.pin_captured] += pin_vel[~self.pin_captured] * dt
        self.pin_pos[self.pin_captured] = self.portal_pos # Поглощенные когерентности застывают в центре

        # 2. ГЕНЕРАЦИЯ СЛАЙМА И ВПРЫСК 120 КОГЕРЕНТНОСТЕЙ (Только от активных точек)
        gaussians = torch.exp(-(dx_shape**2 + dy_shape**2) / node_radius) * (1.0 - cap_mask_f.view(16, 1, 1))
        resting_player_density = torch.max(gaussians, dim=0, keepdim=True)[0].unsqueeze(0)

        if is_real_data and eeg_c0 is not None:
            is_active_1d = (~self.pin_captured).float()
            active_matrix = is_active_1d.unsqueeze(1) * is_active_1d.unsqueeze(0)
            c0_gpu = eeg_c0.to(self.device).float()[:16, :16] * active_matrix
            
            raw_diffs = self.pin_pos.unsqueeze(0) - self.pin_pos.unsqueeze(1)
            directions = raw_diffs / (torch.norm(raw_diffs, dim=2, keepdim=True) + 1e-5)
            
            mid_gx = (pin_gx.unsqueeze(0) + pin_gx.unsqueeze(1)) * 0.5
            mid_gy = (pin_gy.unsqueeze(0) + pin_gy.unsqueeze(1)) * 0.5
            
            dx_grid = self.x_indices.unsqueeze(-1).unsqueeze(-1) - mid_gx.unsqueeze(0).unsqueeze(0)
            dy_grid = self.y_indices.unsqueeze(-1).unsqueeze(-1) - mid_gy.unsqueeze(0).unsqueeze(0)
            
            pair_footprint = torch.exp(-(dx_grid**2 + dy_grid**2) / 8.0) 
            pair_force_x = directions[:, :, 0] * c0_gpu
            pair_force_y = directions[:, :, 1] * c0_gpu
            
            total_force_x = torch.sum(pair_footprint * pair_force_x.unsqueeze(0).unsqueeze(0), dim=(2, 3))
            total_force_y = torch.sum(pair_footprint * pair_force_y.unsqueeze(0).unsqueeze(0), dim=(2, 3))

            self.u[0, 0] += self.player_density[0, 0] * total_force_x * 90.0 * dt
            self.v[0, 0] += self.player_density[0, 0] * total_force_y * 90.0 * dt
            
            force_mag = torch.sqrt(total_force_x**2 + total_force_y**2)
            injection_mask = self.player_density[0, 0] * (force_mag > 0.05).float()
            self.density[0, 0] += injection_mask * torch.clamp(force_mag / 5.0, 0.0, 1.0)
            self.density[0, 1] += injection_mask * torch.clamp(force_mag / 8.0, 0.0, 0.8)
            self.density[0, 2] += injection_mask * torch.clamp(force_mag / 20.0, 0.0, 0.2)

        # 3. ЭСТЕТИКА ПОРТАЛА (Красивый темный водоворот, не притягивает физически)
        goal_gx = (self.portal_pos[0] / self.WIDTH) * self.res
        goal_gy = (self.portal_pos[1] / self.HEIGHT) * self.res
        dx_portal = self.x_indices - goal_gx
        dy_portal = self.y_indices - goal_gy
        dist_portal = torch.sqrt(dx_portal**2 + dy_portal**2) + 1e-5
        
        core_mask = torch.exp(-(dist_portal**2) / ((self.res / self.maze.dim * 0.4)**2))
        self.density[0, 0] -= core_mask * 3.0 * dt
        self.density[0, 1] -= core_mask * 3.0 * dt
        self.density[0, 2] += core_mask * 2.5 * dt 
        self.density = torch.clamp(self.density, 0.0, 1.0)

        # 4. ШАГ НАВЬЕ-СТОКСА И ЭРОЗИЯ СТЕН
        self.u = self.solver.advect(self.u, self.u, self.v, dt)
        self.v = self.solver.advect(self.v, self.u, self.v, dt)
        self.density = self.solver.advect(self.density, self.u, self.v, dt)
        
        # Эрозия (разрушение) стен
        erosion = torch.sqrt(self.u**2 + self.v**2) * 0.15 * dt
        self.wall_density = torch.clamp(self.wall_density - erosion, 0.0, 1.0)
        self.wall_density += (self.orig_obstacles - self.wall_density) * 0.15 * dt
        self.wall_density = torch.clamp(self.wall_density, 0.0, 1.0)
        
        self.u, self.v = self.solver.project(self.u, self.v, self.wall_density)

        if torch.sum(self.player_density) < 1e-4:
            self.player_density.copy_(resting_player_density)
            
        self.player_density = self.solver.advect(self.player_density, self.u, self.v, dt * 1.5)
        self.player_density += (resting_player_density - self.player_density) * 4.5 * dt
        
        # Жесткое отсечение слайма (для визуала, предотвращает наползание)
        hard_walls = (self.wall_density > 0.1).float()
        self.player_density = self.player_density * (1.0 - hard_walls)
        self.player_density = torch.clamp(self.player_density, 0.0, 1.0)

        # 5. КООРДИНАТЫ И ВРАЩЕНИЕ КАМЕРЫ (Слежение по полному центру масс)
        self.player_pos[0] = com[0]
        self.player_pos[1] = com[1]

        u_pad = F.pad(self.u, (1, 1, 1, 1), mode='replicate')
        v_pad = F.pad(self.v, (1, 1, 1, 1), mode='replicate')
        vorticity = 0.5 * (v_pad[:, :, 1:-1, 2:] - v_pad[:, :, 1:-1, :-2]) - 0.5 * (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1])
        
        d_sum = torch.sum(self.player_density) + 1e-6
        self.smooth_vorticity = self.smooth_vorticity * 0.92 + (torch.sum(vorticity * self.player_density) / d_sum).item() * 0.08
        
        if is_real_data:
            self.player_angle += self.smooth_vorticity * dt * self.cfg['vorticity_sensitivity'] * 5.0
            self.player_angle = (self.player_angle + math.pi) % (2.0 * math.pi) - math.pi

        # 6. ПОБЕДА (Когда все когерентности засосало в сингулярность)
        if self.pin_captured.sum().item() == 16:
            self.reset_world()
