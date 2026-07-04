# vortex_physics.py
import torch
import torch.nn.functional as F
import math
import numpy as np
import random
from vortex_fluid import FluidSolver
from vortex_maze import PythonMaze
from vortex_obstacles import init_arena_obstacles
from implicit_config import COORDS_16_X, COORDS_16_Y
from vortex_softbody import update_neighbor_springs, apply_cohesion_constraint


class PhaseVortexArena:
    def __init__(self, device, width, height, res, seed=202607):
        self.device = device
        self.WIDTH, self.HEIGHT = width, height
        self.res = res
        self.current_seed = seed
        self.solver = FluidSolver(res, device)
        
        # === ЦЕНТРАЛИЗОВАННЫЙ СЛОВАРЬ НАСТРОЕК СИМУЛЯЦИИ ===
        self.cfg = {
            'vorticity_sensitivity': 0.15, 
            'fluid_damping': 0.80,
            'eeg_force_scale': 4.50,         
            
            # Взаимодействие со стенами
            'inner_wall_repulsion_scale': 8000.0,   
            'inner_wall_penetration_limit': 0.04,   
            'outer_wall_repulsion_scale': 150000.0, 
            'outer_wall_penetration_limit': 0.001,  
            
            'hide_tiled_labyrinths': True,    
            'show_debug_window': True,       # Окно дебага формы и когерентностей FreeEEG16
            
            # Параметры связи когерентностей и вращения экрана
            'coherence_relative_to_physical': True,  
            'torque_sensitivity_multiplier': 0.0008, 
            
            # === ПАРАМЕТРЫ НАСТРОЙКИ СЖАТИЯ / РАСШИРЕНИЯ ===
            'base_scale': 1.25,              
            'compress_scale_mult': 0.45,     
            'expand_scale_mult': 1.25,       
            
            'base_stiffness': 150.0,          
            'compress_stiffness_add': 250.0,  
            'expand_stiffness_sub': 100.0,    
            
            'base_node_radius': 8.0,         
            'compress_radius_sub': 3.0,      
            'expand_radius_add': 6.0,        
            
            'fluid_div_force': 25.0,         
        }
        self.smooth_vorticity = 0.0
        
        self.u = torch.zeros((1, 1, res, res), device=device)
        self.v = torch.zeros((1, 1, res, res), device=device)
        self.density = torch.zeros((1, 3, res, res), device=device)
        self.player_density = torch.zeros((1, 1, res, res), device=device)
        self.wall_density = torch.zeros((1, 1, res, res), device=device)
        
        self.inner_obstacles = torch.zeros((1, 1, res, res), device=device)
        self.outer_obstacles = torch.zeros((1, 1, res, res), device=device)
        self.inner_wall_density = torch.zeros((1, 1, res, res), device=device)
        
        self.pin_x = torch.tensor(COORDS_16_X, dtype=torch.float32, device=device)
        self.pin_y = torch.tensor(COORDS_16_Y, dtype=torch.float32, device=device)
        
        # Предрасчет базовой матрицы идеальных дистанций (Скелет)
        dx_base = self.pin_x.unsqueeze(0) - self.pin_x.unsqueeze(1)
        dy_base = self.pin_y.unsqueeze(0) - self.pin_y.unsqueeze(1)
        self.base_dist = torch.sqrt(dx_base**2 + dy_base**2)
        
        self.pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
        self.pin_captured = torch.zeros(16, dtype=torch.bool, device=device)
        self.edge_intact = torch.ones(16, dtype=torch.bool, device=device)
        
        self.player_pos = torch.tensor([width/2, height/2], dtype=torch.float32, device=device)
        self.portal_pos = torch.zeros(2, dtype=torch.float32, device=device)
        self.player_angle = 0.0
        self.debug_dot_cov = 1.0
        self.debug_cross_cov = 0.0
        self.screen_size = torch.tensor([width, height], dtype=torch.float32, device=device)
        
        self.eeg_c0_matrix = torch.zeros((16, 16), device=device)
        
        self.y_indices, self.x_indices = torch.meshgrid(
            torch.arange(res, device=device, dtype=torch.float32),
            torch.arange(res, device=device, dtype=torch.float32), indexing='ij'
        )
        
        self.rune_zones = []
        self.reset_world()
        
    def init_obstacles(self):
        return init_arena_obstacles(self.solver.grid_x, self.solver.grid_y, self.maze.grid, self.res, self.device)

    def reset_world(self):
        self.maze = PythonMaze(11, seed=self.current_seed)
        dim = self.maze.dim
        goal_rows, goal_cols = np.where(self.maze.grid == 2)
        
        self.goal_cell = (goal_cols[0], goal_rows[0]) if len(goal_rows) > 0 else (dim - 2, dim - 2)
        
        self.inner_obstacles, self.outer_obstacles = self.init_obstacles()
        self.orig_obstacles = self.inner_obstacles + self.outer_obstacles
        
        self.cell_w = (self.WIDTH * 0.8) / self.maze.dim
        spawn_pos = (self.WIDTH * 0.1) + 1.5 * self.cell_w
        
        self.portal_pos[0] = (self.WIDTH * 0.1) + (self.goal_cell[0] + 0.5) * self.cell_w
        self.portal_pos[1] = (self.HEIGHT * 0.1) + (self.goal_cell[1] + 0.5) * self.cell_w
        
        self.player_pos.copy_(torch.tensor([spawn_pos, spawn_pos], dtype=torch.float32, device=self.device))
        self.player_angle = 0.0
        
        self.pin_captured.zero_()
        self.edge_intact.fill_(True)
        self.pin_pos[:, 0] = spawn_pos + self.pin_x * 1.5
        self.pin_pos[:, 1] = spawn_pos + self.pin_y * 1.5
        
        self.u.zero_()
        self.v.zero_()
        self.density.zero_()
        self.player_density.zero_()
        
        self.inner_wall_density.copy_(self.inner_obstacles)
        self.wall_density.copy_(self.orig_obstacles)
        self.smooth_vorticity = 0.0
        self.eeg_c0_matrix.zero_()

        # Инициализация Рунических Зон в пустых ячейках лабиринта
        self.rune_zones = []
        empty_cells = []
        for r in range(1, dim - 1):
            for c in range(1, dim - 1):
                if self.maze.grid[r, c] == 0:
                    if (c, r) != (1, 1) and (c, r) != self.goal_cell:
                        empty_cells.append((c, r))
                        
        rng = random.Random(self.current_seed)
        rng.shuffle(empty_cells)
        
        num_zones = min(3, len(empty_cells))
        for i in range(num_zones):
            cz, rz = empty_cells[i]
            zx = (self.WIDTH * 0.1) + (cz + 0.5) * self.cell_w
            zy = (self.HEIGHT * 0.1) + (rz + 0.5) * self.cell_w
            self.rune_zones.append({
                'cell': (cz, rz),
                'pos': torch.tensor([zx, zy], dtype=torch.float32, device=self.device),
                'radius': self.cell_w * 0.5,
                'charge': 0.0,
                'completed': False,
                'classification': "Pending",
                'telemetry': {
                    'deformation': [],
                    'vorticity': [],
                    'acceleration': [],
                    'angles': [],
                    'vel_mags': []
                },
                'last_vel': torch.zeros(2, dtype=torch.float32, device=self.device)
            })

    def _classify_control_style(self, telemetry):
        if len(telemetry['deformation']) < 5:
            return "Gamepad"
            
        defs = np.array(telemetry['deformation'])
        angles = np.array(telemetry['angles'])
        accels = np.array(telemetry['acceleration'])
        vel_mags = np.array(telemetry['vel_mags'])
        
        pi_4 = np.pi / 4.0
        angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi
        grid_devs = np.minimum(np.abs(angles % pi_4), pi_4 - np.abs(angles % pi_4))
        mean_grid_dev = np.mean(grid_devs)
        accel_jitter = np.std(accels)
        mean_def = np.mean(defs)
        
        if accel_jitter < 3.5:
            return "AI (Autopilot)"
        if mean_grid_dev < 0.06:
            return "Keyboard"
        if mean_def > 14.0:
            return "Neuroslime (Direct EEG)"
        if accel_jitter > 28.0:
            return "Neurogamepad (EEG-Stick)"
            
        return "Gamepad"

    def step(self, dt, time_sec, eeg_c0_spectrum, eeg_vx, eeg_vy, eeg_tq, is_real_data, compression, scale_factor, eeg_freqs=None):
        if torch.is_tensor(eeg_vx): eeg_vx = eeg_vx.item()
        if torch.is_tensor(eeg_vy): eeg_vy = eeg_vy.item()
        if torch.is_tensor(eeg_tq): eeg_tq = eeg_tq.item()

        # === 1. АСИММЕТРИЧНЫЙ КОНТИНУУМ (ДИНАМИЧЕСКИЕ ПАРАМЕТРЫ ИЗ CFG) ===
        blend = max(-1.0, min(1.0, compression))
        
        if blend < 0.0:
            scale = self.cfg['base_scale'] - blend * self.cfg['expand_scale_mult']
            stiffness = self.cfg['base_stiffness'] + blend * self.cfg['expand_stiffness_sub']
            node_radius = self.cfg['base_node_radius'] - blend * self.cfg['expand_radius_add']
        else:
            scale = self.cfg['base_scale'] - blend * self.cfg['compress_scale_mult']
            stiffness = self.cfg['base_stiffness'] + blend * self.cfg['compress_stiffness_add']
            node_radius = self.cfg['base_node_radius'] - blend * self.cfg['compress_radius_sub']

        self.u = torch.nan_to_num(self.u, nan=0.0) * 0.99
        self.v = torch.nan_to_num(self.v, nan=0.0) * 0.99
        self.density = torch.nan_to_num(self.density, nan=0.0) * 0.985
        self.u = torch.clamp(self.u, -250.0, 250.0)
        self.v = torch.clamp(self.v, -250.0, 250.0)
        
        # Динамика стен
        self.inner_wall_density = self.solver.advect(self.inner_wall_density, self.u, self.v, dt * 0.05)
        erosion = torch.sqrt(self.u**2 + self.v**2) * 0.005 * dt
        self.inner_wall_density = torch.clamp(self.inner_wall_density - erosion, 0.0, 1.0)
        self.inner_wall_density += (self.inner_obstacles - self.inner_wall_density) * 1.8 * dt
        self.inner_wall_density = torch.clamp(self.inner_wall_density, 0.0, 1.0)
        self.wall_density = torch.clamp(self.inner_wall_density + self.outer_obstacles, 0.0, 1.0)

        if self.pin_captured.all():
            self.reset_world()
            return
            
        com = self.pin_pos.mean(dim=0)
        
        # === 2. ЧЕСТНОЕ МАТЕМАТИЧЕСКОЕ ВРАЩЕНИЕ С ЗАЩИТОЙ ОТ СМЯТИЯ ОБ СТЕНУ ===
        actual_local_x = self.pin_pos[:, 0] - com[0]
        actual_local_y = self.pin_pos[:, 1] - com[1]
        
        cross_cov = torch.sum(self.pin_x * actual_local_y - self.pin_y * actual_local_x)
        dot_cov = torch.sum(self.pin_x * actual_local_x + self.pin_y * actual_local_y) + 1e-5
        
        self.debug_cross_cov = cross_cov.item()
        self.debug_dot_cov = dot_cov.item()
        
        raw_angle = torch.atan2(cross_cov, dot_cov).item()
        
        # Находим кратчайшую разницу углов
        angle_diff = (raw_angle - self.player_angle + math.pi) % (2 * math.pi) - math.pi
        
        # Если скачок угла за 1 кадр аномально велик (> 1.2 рад / ~68 град), это математический артефакт
        # смятия/инверсии матрицы при ударе об стену, а не реальный физический поворот.
        if abs(angle_diff) < 1.2:
            # Плавная фильтрация с ограничением максимальной скорости вращения (6.0 рад/с)
            max_step = 6.0 * dt
            clamped_diff = max(-max_step, min(max_step, angle_diff))
            self.player_angle += clamped_diff * 0.30
            
        cos_p, sin_p = math.cos(self.player_angle), math.sin(self.player_angle)
        
        # Шаблон идеальной геометрии FreeEEG16, повернутый на стабильный угол камеры
        ideal_x = self.pin_x * cos_p - self.pin_y * sin_p
        ideal_y = self.pin_x * sin_p + self.pin_y * cos_p
        
        ideal_x_scaled = ideal_x * scale
        ideal_y_scaled = ideal_y * scale
        ideal_pos = com.unsqueeze(0) + torch.stack([ideal_x_scaled, ideal_y_scaled], dim=1)
        
        f_spring_x = (ideal_pos[:, 0] - self.pin_pos[:, 0]) * stiffness
        f_spring_y = (ideal_pos[:, 1] - self.pin_pos[:, 1]) * stiffness
        f_spring = (ideal_pos - self.pin_pos) * stiffness

        # Вычисляем взаимные силы натяжения между соседними нодами
        f_neighbor, self.edge_intact = update_neighbor_springs(
            self.pin_pos, ideal_pos, self.edge_intact, blend, self.device
        )
        
        # === ГЕОМЕТРИЯ СЕТКИ ===
        pin_gx = torch.remainder((self.pin_pos[:, 0] / self.WIDTH) * self.res, self.res)
        pin_gy = torch.remainder((self.pin_pos[:, 1] / self.HEIGHT) * self.res, self.res)
        
        dx_shape = torch.remainder(self.x_indices.unsqueeze(0) - pin_gx.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
        dy_shape = torch.remainder(self.y_indices.unsqueeze(0) - pin_gy.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
        is_active_1d = (~self.pin_captured).float()
        
        node_influence = torch.exp(-(dx_shape**2 + dy_shape**2) / node_radius) * is_active_1d.reshape(16, 1, 1)
        
        # Нормализуем влияние каждой ноды
        node_influence_sum = torch.sum(node_influence, dim=(1, 2), keepdim=True) + 1e-8
        node_influence_normalized = node_influence / node_influence_sum
        
        pin_uv_raw = (self.pin_pos / self.screen_size) * 2.0 - 1.0
        pin_uv = torch.remainder(pin_uv_raw + 1.0, 2.0) - 1.0
        sampled_u = F.grid_sample(self.u, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
        sampled_v = F.grid_sample(self.v, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
        fluid_vel = torch.stack([sampled_u, sampled_v], dim=1) * 80.0 
        
        # Чтение коллизий стен (Градиенты)
        w_val_inner = F.grid_sample(self.inner_wall_density, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_pad_inner = F.pad(self.inner_wall_density, (1, 1, 1, 1), mode='circular')
        grad_x_inner = 0.5 * (w_pad_inner[:, :, 1:-1, 2:] - w_pad_inner[:, :, 1:-1, :-2])
        grad_y_inner = 0.5 * (w_pad_inner[:, :, 2:, 1:-1] - w_pad_inner[:, :, :-2, 1:-1])
        w_gx_inner = F.grid_sample(grad_x_inner, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_gy_inner = F.grid_sample(grad_y_inner, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        grad_norm_inner = torch.sqrt(w_gx_inner**2 + w_gy_inner**2) + 1e-5
        dir_out_x_inner = -w_gx_inner / grad_norm_inner
        dir_out_y_inner = -w_gy_inner / grad_norm_inner
        f_wall_inner = torch.stack([dir_out_x_inner * w_val_inner * self.cfg.get('inner_wall_repulsion_scale', 8000.0), 
                                    dir_out_y_inner * w_val_inner * self.cfg.get('inner_wall_repulsion_scale', 8000.0)], dim=1)
        
        w_val_outer = F.grid_sample(self.outer_obstacles, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_pad_outer = F.pad(self.outer_obstacles, (1, 1, 1, 1), mode='circular')
        grad_x_outer = 0.5 * (w_pad_outer[:, :, 1:-1, 2:] - w_pad_outer[:, :, 1:-1, :-2])
        grad_y_outer = 0.5 * (w_pad_outer[:, :, 2:, 1:-1] - w_pad_outer[:, :, :-2, 1:-1])
        w_gx_outer = F.grid_sample(grad_x_outer, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_gy_outer = F.grid_sample(grad_y_outer, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        grad_norm_outer = torch.sqrt(w_gx_outer**2 + w_gy_outer**2) + 1e-5
        dir_out_x_outer = -w_gx_outer / grad_norm_outer
        dir_out_y_outer = -w_gy_outer / grad_norm_outer
        f_wall_outer = torch.stack([dir_out_x_outer * w_val_outer * self.cfg.get('outer_wall_repulsion_scale', 150000.0), 
                                    dir_out_y_outer * w_val_outer * self.cfg.get('outer_wall_repulsion_scale', 150000.0)], dim=1)
        f_wall = f_wall_inner + f_wall_outer
        
        dx_p = torch.remainder(self.pin_pos[:, 0] - self.portal_pos[0] + self.WIDTH/2, self.WIDTH) - self.WIDTH/2
        dy_p = torch.remainder(self.pin_pos[:, 1] - self.portal_pos[1] + self.HEIGHT/2, self.HEIGHT) - self.HEIGHT/2
        self.pin_captured = self.pin_captured | ((torch.sqrt(dx_p**2 + dy_p**2)) < self.cell_w * 0.35)
        
        # === 3. ФИЗИКА КОГЕРЕНТНОСТЕЙ (ДУНОВЕНИЕ) ===
        active_matrix = is_active_1d.unsqueeze(1) * is_active_1d.unsqueeze(0)

        dist_local = torch.sqrt(actual_local_x**2 + actual_local_y**2) + 1e-5
        tangent_x = -actual_local_y / dist_local
        tangent_y = actual_local_x / dist_local

        if is_real_data and eeg_c0_spectrum is not None:
            c0_gpu = eeg_c0_spectrum[:16, :16, :] * active_matrix.unsqueeze(-1)
            c0_total = torch.sum(c0_gpu, dim=2) 
            self.eeg_c0_matrix.copy_(c0_total)
            
            if self.cfg.get('coherence_relative_to_physical', True):
                dx_ideal = ideal_x.unsqueeze(1) - ideal_x.unsqueeze(0)
                dy_ideal = ideal_y.unsqueeze(1) - ideal_y.unsqueeze(0)
            else:
                dx_ideal = self.pin_x.unsqueeze(1) - self.pin_x.unsqueeze(0)
                dy_ideal = self.pin_y.unsqueeze(1) - self.pin_y.unsqueeze(0)
            
            force_multiplier = 35.0
            node_bci_force_x = torch.sum(c0_total * dx_ideal, dim=1) * force_multiplier
            node_bci_force_y = torch.sum(c0_total * dy_ideal, dim=1) * force_multiplier
            
            # Вращение от BCI
            node_bci_force_x += tangent_x * eeg_tq * 80.0
            node_bci_force_y += tangent_y * eeg_tq * 80.0
            
            bci_propulsion = torch.stack([node_bci_force_x, node_bci_force_y], dim=1)
            bci_mag = torch.sqrt(torch.sum(node_bci_force_x)**2 + torch.sum(node_bci_force_y)**2).item() / 100.0
            node_coherence = torch.sum(torch.abs(c0_total), dim=1)
        else:
            world_vx = eeg_vx * cos_p + eeg_vy * sin_p
            world_vy = -eeg_vx * sin_p + eeg_vy * cos_p
            bci_mag = math.sqrt(world_vx**2 + world_vy**2)
            
            node_bci_force_x = torch.tensor(world_vx, device=self.device).repeat(16) * 120.0 + tangent_x * eeg_tq * 60.0
            node_bci_force_y = torch.tensor(world_vy, device=self.device).repeat(16) * 120.0 + tangent_y * eeg_tq * 60.0
            bci_propulsion = torch.stack([node_bci_force_x, node_bci_force_y], dim=1)
            node_coherence = torch.full((16,), bci_mag * 0.5 + 0.1, device=self.device)
            self.eeg_c0_matrix.zero_()

        # === ИНЖЕКЦИЯ СИЛ В ЖИДКОСТЬ ===
        eeg_react_x = -torch.sum(node_influence_normalized * node_bci_force_x.reshape(16, 1, 1), dim=0) * 0.20
        eeg_react_y = -torch.sum(node_influence_normalized * node_bci_force_y.reshape(16, 1, 1), dim=0) * 0.20
        
        f_spring_clamped_x = torch.clamp(f_spring_x, -1200.0, 1200.0)
        f_spring_clamped_y = torch.clamp(f_spring_y, -1200.0, 1200.0)
        
        f_total_elastic_x = f_spring_clamped_x + f_neighbor[:, 0]
        f_total_elastic_y = f_spring_clamped_y + f_neighbor[:, 1]
        
        spring_force_grid_x = torch.sum(node_influence_normalized * f_total_elastic_x.reshape(16, 1, 1), dim=0)
        spring_force_grid_y = torch.sum(node_influence_normalized * f_total_elastic_y.reshape(16, 1, 1), dim=0)

        bci_force_grid_x = torch.sum(node_influence_normalized * node_bci_force_x.reshape(16, 1, 1), dim=0)
        bci_force_grid_y = torch.sum(node_influence_normalized * node_bci_force_y.reshape(16, 1, 1), dim=0)

        self.u[0, 0] += (eeg_react_x + bci_force_grid_x * 1.5 + spring_force_grid_x * 0.5) * dt
        self.v[0, 0] += (eeg_react_y + bci_force_grid_y * 1.5 + spring_force_grid_y * 0.5) * dt

        # === 4. КИНЕМАТИКА (ПОЛНОСТЬЮ НА СИЛАХ ЖИДКОСТИ) ===
        pin_vel = fluid_vel * 0.85 + f_wall

        # Скольжение вдоль стен
        dot_inner = pin_vel[:, 0] * dir_out_x_inner + pin_vel[:, 1] * dir_out_y_inner
        limit_inner = self.cfg.get('inner_wall_penetration_limit', 0.04)
        blocking_inner = torch.clamp(w_val_inner / limit_inner, 0.0, 1.0)
        moving_into_inner = dot_inner < 0
        pin_vel[:, 0] -= torch.where(moving_into_inner, dot_inner * dir_out_x_inner * blocking_inner, torch.zeros_like(pin_vel[:, 0]))
        pin_vel[:, 1] -= torch.where(moving_into_inner, dot_inner * dir_out_y_inner * blocking_inner, torch.zeros_like(pin_vel[:, 1]))
        
        dot_outer = pin_vel[:, 0] * dir_out_x_outer + pin_vel[:, 1] * dir_out_y_outer
        limit_outer = self.cfg.get('outer_wall_penetration_limit', 0.001)
        blocking_outer = torch.clamp(w_val_outer / limit_outer, 0.0, 1.0)
        moving_into_outer = dot_outer < 0
        pin_vel[:, 0] -= torch.where(moving_into_outer, dot_outer * dir_out_x_outer * blocking_outer, torch.zeros_like(pin_vel[:, 0]))
        pin_vel[:, 1] -= torch.where(moving_into_outer, dot_outer * dir_out_y_outer * blocking_outer, torch.zeros_like(pin_vel[:, 1]))
        
        pin_vel = torch.clamp(pin_vel, -180.0, 180.0)
        self.pin_pos[~self.pin_captured] += pin_vel[~self.pin_captured] * dt
        
        # Геометрический ограничитель PBD
        self.pin_pos = apply_cohesion_constraint(
            self.pin_pos, ideal_pos, self.pin_captured, scale, blend
        )
        
        # === 5. ПОСТ-ШАГОВОЕ ВЫТАЛКИВАНИЕ ИЗ СТЕН ===
        pin_uv_raw_post = (self.pin_pos / self.screen_size) * 2.0 - 1.0
        pin_uv_post = torch.remainder(pin_uv_raw_post + 1.0, 2.0) - 1.0
        
        w_val_inner_post = F.grid_sample(self.inner_wall_density, pin_uv_post.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_val_outer_post = F.grid_sample(self.outer_obstacles, pin_uv_post.view(1, 1, 16, 2), align_corners=True).squeeze()
        
        pushed_inner = w_val_inner_post > limit_inner
        if pushed_inner.any():
            push_dist_in = (w_val_inner_post - limit_inner) * 15.0
            self.pin_pos[pushed_inner, 0] += dir_out_x_inner[pushed_inner] * push_dist_in[pushed_inner]
            self.pin_pos[pushed_inner, 1] += dir_out_y_inner[pushed_inner] * push_dist_in[pushed_inner]
            
        pushed_outer = w_val_outer_post > limit_outer
        if pushed_outer.any():
            push_dist_out = (w_val_outer_post - limit_outer) * 25.0
            self.pin_pos[pushed_outer, 0] += dir_out_x_outer[pushed_outer] * push_dist_out[pushed_outer]
            self.pin_pos[pushed_outer, 1] += dir_out_y_outer[pushed_outer] * push_dist_out[pushed_outer]
            
        virtual_portal_pos = torch.stack([self.pin_pos[:, 0] - dx_p, self.pin_pos[:, 1] - dy_p], dim=1)
        self.pin_pos[self.pin_captured] = virtual_portal_pos[self.pin_captured] 

        # === 6. ОБНОВЛЕНИЕ ЖИДКОСТИ И РЕНДЕР ===
        self.player_density = self.solver.advect(self.player_density, self.u, self.v, dt * 0.4)
        self.density = self.solver.advect(self.density, self.u, self.v, dt * 0.4)

        resting_player_density = torch.max(node_influence, dim=0, keepdim=True)[0].unsqueeze(0)

        smoke_grid = torch.sum(node_influence * node_coherence.reshape(16, 1, 1), dim=0)
        injection_mask = smoke_grid * (smoke_grid > 0.05).float()
        self.density[0, 0] += injection_mask * min(1.0, max(0.0, bci_mag / 2.0)) * dt * 45.0
        self.density[0, 1] += injection_mask * min(0.8, max(0.0, bci_mag / 3.0)) * dt * 45.0
        self.density[0, 2] += injection_mask * min(0.2, max(0.0, bci_mag / 10.0)) * dt * 45.0

        # Воронка портала
        goal_gx = (self.portal_pos[0] / self.WIDTH) * self.res
        goal_gy = (self.portal_pos[1] / self.HEIGHT) * self.res
        dx_portal = torch.remainder(self.x_indices - goal_gx + self.res/2, self.res) - self.res/2
        dy_portal = torch.remainder(self.y_indices - goal_gy + self.res/2, self.res) - self.res/2
        core_mask = torch.exp(-((dx_portal**2 + dy_portal**2)) / ((self.res / self.maze.dim * 0.4)**2))
        
        self.density[0, 0] -= core_mask * 3.0 * dt
        self.density[0, 1] -= core_mask * 3.0 * dt
        self.density[0, 2] += core_mask * 2.5 * dt 
        self.density = torch.clamp(self.density, 0.0, 1.0)

        self.u = self.solver.advect(self.u, self.u, self.v, dt)
        self.v = self.solver.vvect(self.v, self.u, self.v, dt) if hasattr(self.solver, 'vvect') else self.solver.advect(self.v, self.u, self.v, dt)
        self.density = self.solver.advect(self.density, self.u, self.v, dt)
        
        erosion = torch.sqrt(self.u**2 + self.v**2) * 0.15 * dt
        self.wall_density = torch.clamp(self.wall_density - erosion, 0.0, 1.0)
        self.wall_density += (self.orig_obstacles - self.wall_density) * 0.15 * dt
        self.wall_density = torch.clamp(self.wall_density, 0.0, 1.0)
        
        target_div_grid = -blend * self.cfg['fluid_div_force'] * torch.sum(node_influence_normalized, dim=0, keepdim=True).unsqueeze(0)
        
        self.u, self.v = self.solver.project(self.u, self.v, self.wall_density, target_div_grid)

        self.density = self.density * (self.player_density * 0.95 + 0.05)
        if torch.sum(self.player_density) < 1e-4: self.player_density.copy_(resting_player_density)
            
        self.player_density = self.solver.advect(self.player_density, self.u, self.v, dt * 1.5)
        self.player_density += (resting_player_density - self.player_density) * 4.5 * dt
        self.player_density = torch.clamp(self.player_density * (1.0 - (self.wall_density > 0.1).float()), 0.0, 1.0)

        self.player_pos[0], self.player_pos[1] = com[0], com[1]

        u_pad, v_pad = F.pad(self.u, (1, 1, 1, 1), mode='circular'), F.pad(self.v, (1, 1, 1, 1), mode='circular')
        vorticity = 0.5 * (v_pad[:, :, 1:-1, 2:] - v_pad[:, :, 1:-1, :-2]) - 0.5 * (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1])
        self.smooth_vorticity = self.smooth_vorticity * 0.90 + (torch.sum(vorticity * self.player_density) / (torch.sum(self.player_density) + 1e-6)).item() * 0.10

        # === АКТИВАЦИЯ И ТЕЛЕМЕТРИЯ РУНИЧЕСКИХ ЗОН ===
        for zone in self.rune_zones:
            if zone['completed']:
                continue
                
            dx_z = self.player_pos[0] - zone['pos'][0]
            dy_z = self.player_pos[1] - zone['pos'][1]
            dist_z = torch.sqrt(dx_z**2 + dy_z**2).item()
            
            if dist_z < zone['radius'] * 1.2:
                local_energy = abs(self.smooth_vorticity) * 2.5 + 0.15
                zone['charge'] = min(1.0, zone['charge'] + local_energy * dt * 0.4)
                
                active_vels = pin_vel[~self.pin_captured]
                v_com = active_vels.mean(dim=0) if active_vels.numel() > 0 else torch.zeros(2, device=self.device)
                v_mag = torch.norm(v_com).item()
                
                accel = torch.norm(v_com - zone['last_vel']).item() / (dt + 1e-5)
                zone['last_vel'].copy_(v_com)
                
                node_dists = torch.norm(self.pin_pos - self.player_pos, dim=1)
                deform = torch.std(node_dists).item()
                
                zone['telemetry']['deformation'].append(float(deform))
                zone['telemetry']['vorticity'].append(float(self.smooth_vorticity))
                zone['telemetry']['acceleration'].append(float(accel))
                zone['telemetry']['angles'].append(float(math.atan2(v_com[1].item(), v_com[0].item())))
                zone['telemetry']['vel_mags'].append(float(v_mag))
                
                if zone['charge'] >= 1.0:
                    zone['completed'] = True
                    zone['classification'] = self._classify_control_style(zone['telemetry'])
            else:
                zone['charge'] = max(0.0, zone['charge'] - dt * 0.1)

        if self.pin_captured.sum().item() == 16:
            self.reset_world()
