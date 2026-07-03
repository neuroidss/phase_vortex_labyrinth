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
    def __init__(self, device, width, height, res, seed=202607):
        self.device = device
        self.WIDTH, self.HEIGHT = width, height
        self.res = res
        self.current_seed = seed
        self.solver = FluidSolver(res, device)
        
        # Настройки баланса физики, удержания формы и взаимодействия со стенами
        self.cfg = {
            'vorticity_sensitivity': 0.15, 
            'fluid_damping': 0.80,
            'spring_stiffness': 1500.0,        # Базовая жесткость каркаса слайма
            'eeg_force_scale': 4.50,         # Масштаб BCI-сил инжекции
            'manual_force_scale': 600.0,     # Масштаб ручного управления (клавиатура/геймпад)
            'wall_repulsion_scale': 4000.0,   # Сила упругого отталкивания от стен
            'wall_penetration_limit': 0.04,   # Порог плотности стены, при котором она полностью непроходима (sliding collision)
        }
        self.smooth_vorticity = 0.0
        
        self.u = torch.zeros((1, 1, res, res), device=device)
        self.v = torch.zeros((1, 1, res, res), device=device)
        self.density = torch.zeros((1, 3, res, res), device=device)
        self.player_density = torch.zeros((1, 1, res, res), device=device)
        self.wall_density = torch.zeros((1, 1, res, res), device=device)
        
        self.pin_x = torch.tensor(COORDS_16_X, dtype=torch.float32, device=device)
        self.pin_y = torch.tensor(COORDS_16_Y, dtype=torch.float32, device=device)
        
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
        obstacles = torch.zeros((1, 1, self.res, self.res), device=self.device)
        dim = self.maze.dim
        goal_rows, goal_cols = np.where(self.maze.grid == 2)
        self.goal_cell = (goal_cols[0], goal_rows[0]) if len(goal_rows) > 0 else (dim - 2, dim - 2)
            
        maze_tensor = torch.tensor(self.maze.grid, dtype=torch.float32, device=self.device)
        scale_limit = 0.8
        
        col_idx = ((self.solver.grid_x + scale_limit) / (2.0 * scale_limit) * dim).long()
        row_idx = ((self.solver.grid_y + scale_limit) / (2.0 * scale_limit) * dim).long()
        
        within_bounds = (torch.abs(self.solver.grid_x) <= scale_limit) & (torch.abs(self.solver.grid_y) <= scale_limit)
        
        valid_cols = col_idx[within_bounds].clamp(0, dim - 1)
        valid_rows = row_idx[within_bounds].clamp(0, dim - 1)
        obstacles[0, 0, within_bounds] = torch.where(maze_tensor[valid_rows, valid_cols] == 1.0, 1.0, 0.0)
        
        return obstacles

    def reset_world(self):
        self.maze = PythonMaze(11, seed=self.current_seed)
        self.orig_obstacles = self.init_obstacles()
        
        self.cell_w = (self.WIDTH * 0.8) / self.maze.dim
        spawn_pos = (self.WIDTH * 0.1) + 1.5 * self.cell_w
        
        self.portal_pos[0] = (self.WIDTH * 0.1) + (self.goal_cell[0] + 0.5) * self.cell_w
        self.portal_pos[1] = (self.HEIGHT * 0.1) + (self.goal_cell[1] + 0.5) * self.cell_w
        
        self.player_pos.copy_(torch.tensor([spawn_pos, spawn_pos], dtype=torch.float32, device=self.device))
        self.player_angle = 0.0
        
        self.pin_captured.zero_()
        self.pin_pos[:, 0] = spawn_pos + self.pin_x * 1.5
        self.pin_pos[:, 1] = spawn_pos + self.pin_y * 1.5
        
        self.u.zero_()
        self.v.zero_()
        self.density.zero_()
        self.player_density.zero_()
        self.wall_density.copy_(self.orig_obstacles)
        self.smooth_vorticity = 0.0

    def step(self, dt, time_sec, eeg_c0_spectrum, eeg_vx, eeg_vy, eeg_tq, is_real_data, compression, scale_factor, eeg_freqs=None):
        # Алгоритм масштабирования под биполярную шкалу [-1.0 (разжат), 0.0 (нейтрален), 1.0 (сжат)]
        if compression > 0.0:
            scale = 1.25 - compression * 1.10          # Сжатие: стягиваем в плотное ядро (до 0.15)
            node_radius = 8.0 - compression * 4.0
            stiffness = 150.0 + compression * 2350.0   # Сверхвысокая жесткость до 2500.0 для отклика без задержек
        else:
            scale = 1.25 - compression * 5.25          # Разжатие: расширяем в масштабный невод (до 6.5)
            node_radius = 8.0 - compression * 32.0
            stiffness = 150.0 + compression * 135.0    # Мягкая просадка упругости для волновых огибаний

        self.u = torch.nan_to_num(self.u, nan=0.0) * 0.99
        self.v = torch.nan_to_num(self.v, nan=0.0) * 0.99
        self.density = torch.nan_to_num(self.density, nan=0.0) * 0.985
        self.u = torch.clamp(self.u, -250.0, 250.0)
        self.v = torch.clamp(self.v, -250.0, 250.0)
        
        # === ДИНАМИКА СТЕН ===
        self.wall_density = self.solver.advect(self.wall_density, self.u, self.v, dt * 0.05)
        erosion = torch.sqrt(self.u**2 + self.v**2) * 0.005 * dt
        self.wall_density = torch.clamp(self.wall_density - erosion, 0.0, 1.0)
        self.wall_density += (self.orig_obstacles - self.wall_density) * 1.8 * dt
        self.wall_density = torch.clamp(self.wall_density, 0.0, 1.0)

        if self.pin_captured.all():
            self.reset_world()
            return
            
        com = self.pin_pos.mean(dim=0)
        
        # ТОРОИДАЛЬНОЕ СЧИТЫВАНИЕ
        pin_uv_raw = (self.pin_pos / self.screen_size) * 2.0 - 1.0
        pin_uv = torch.remainder(pin_uv_raw + 1.0, 2.0) - 1.0
        
        sampled_u = F.grid_sample(self.u, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
        sampled_v = F.grid_sample(self.v, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
        
        fluid_vel = torch.stack([sampled_u, sampled_v], dim=1) * 80.0 
        
        cos_p, sin_p = math.cos(self.player_angle), math.sin(self.player_angle)
        dx_ideal = self.pin_x * cos_p - self.pin_y * sin_p
        dy_ideal = self.pin_x * sin_p + self.pin_y * cos_p
        ideal_pos = com.unsqueeze(0) + torch.stack([dx_ideal, dy_ideal], dim=1) * scale
        
        # Вычисление силы упругости каркаса
        f_spring = (ideal_pos - self.pin_pos) * stiffness
        
        pin_gx = torch.remainder((self.pin_pos[:, 0] / self.WIDTH) * self.res, self.res)
        pin_gy = torch.remainder((self.pin_pos[:, 1] / self.HEIGHT) * self.res, self.res)
        
        # Тороидальные градиенты для инъекции силы упругости в сетку жидкости
        dx_shape = torch.remainder(self.x_indices.unsqueeze(0) - pin_gx.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
        dy_shape = torch.remainder(self.y_indices.unsqueeze(0) - pin_gy.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
        node_footprint = torch.exp(-(dx_shape**2 + dy_shape**2) / 6.0) * (~self.pin_captured).float().reshape(16, 1, 1)
        
        # Предохранительное ограничение силы упругости для защиты сетки жидкости от взрыва при резком сжатии
        f_spring_clamped = torch.clamp(f_spring, -1500.0, 1500.0)
        spring_force_grid_x = torch.sum(node_footprint * f_spring_clamped[:, 0].reshape(16, 1, 1), dim=0)
        spring_force_grid_y = torch.sum(node_footprint * f_spring_clamped[:, 1].reshape(16, 1, 1), dim=0)
        self.u[0, 0] += spring_force_grid_x * dt
        self.v[0, 0] += spring_force_grid_y * dt
        
        w_val = F.grid_sample(self.wall_density, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_pad = F.pad(self.wall_density, (1, 1, 1, 1), mode='circular')
        grad_x = 0.5 * (w_pad[:, :, 1:-1, 2:] - w_pad[:, :, 1:-1, :-2])
        grad_y = 0.5 * (w_pad[:, :, 2:, 1:-1] - w_pad[:, :, :-2, 1:-1])
        w_gx = F.grid_sample(grad_x, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_gy = F.grid_sample(grad_y, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        
        grad_norm = torch.sqrt(w_gx**2 + w_gy**2) + 1e-5
        dir_out_x = -w_gx / grad_norm
        dir_out_y = -w_gy / grad_norm
        
        # Получаем силу упругого отталкивания от стен из конфигурационного словаря
        repulsion_scale = self.cfg.get('wall_repulsion_scale', 4000.0)
        f_wall = torch.stack([
            dir_out_x * w_val * repulsion_scale, 
            dir_out_y * w_val * repulsion_scale
        ], dim=1)
        
        # ТОРОИДАЛЬНЫЙ ПОРТАЛ
        dx_p = torch.remainder(self.pin_pos[:, 0] - self.portal_pos[0] + self.WIDTH/2, self.WIDTH) - self.WIDTH/2
        dy_p = torch.remainder(self.pin_pos[:, 1] - self.portal_pos[1] + self.HEIGHT/2, self.HEIGHT) - self.HEIGHT/2
        dist_to_portal = torch.sqrt(dx_p**2 + dy_p**2)
        
        capture_radius = self.cell_w * 0.35
        self.pin_captured = self.pin_captured | (dist_to_portal < capture_radius)
        cap_mask_f = self.pin_captured.float().unsqueeze(1)
        
        # --- ФИЗИКА КОМПРЕССИИ: ТРАНСЛЯЦИЯ ВЕКТОРОВ НАПРЯМУЮ НА НОДЫ ДЛЯ МГНОВЕННОГО ОТКЛИКА ---
        vx_t = torch.as_tensor(eeg_vx, dtype=torch.float32, device=self.device)
        vy_t = torch.as_tensor(eeg_vy, dtype=torch.float32, device=self.device)
        gamepad_vel = torch.stack([vx_t.repeat(16), vy_t.repeat(16)], dim=1) * 220.0
        
        blend = max(0.0, min(1.0, compression))  # Смешиваем в диапазоне [0.0 (нейтраль) ... 1.0 (сжатие)]
        
        # Сила упругости, транслируемая напрямую в скорость нод для мгновенного сжатия
        v_spring = (ideal_pos - self.pin_pos) * 12.0
        
        # При сжатии замещаем гидродинамику прямой кинематической трансляцией стиков и упругого сжатия
        pin_vel = (1.0 - blend) * fluid_vel * 0.9 + blend * (gamepad_vel + v_spring) + f_wall
        
        # === МАТЕМАТИЧЕСКИ ГЕРМЕТИЧНАЯ СКОЛЬЗЯЩАЯ ПРОЕКЦИЯ КОЛЛИЗИЙ (COLLISION RESOLUTION) ===
        # Проектируем вектор скорости по нормали к стене, гася проникающий импульс
        dot_product = pin_vel[:, 0] * dir_out_x + pin_vel[:, 1] * dir_out_y
        
        # Плотность стены преодолела порог непроходимости
        limit = self.cfg.get('wall_penetration_limit', 0.04)
        blocking_factor = torch.clamp(w_val / limit, 0.0, 1.0)
        
        moving_into_wall = dot_product < 0
        
        # Обнуляем только ту часть вектора, которая пытается двигаться сквозь преграду
        pin_vel[:, 0] = torch.where(moving_into_wall, pin_vel[:, 0] - dot_product * dir_out_x * blocking_factor, pin_vel[:, 0])
        pin_vel[:, 1] = torch.where(moving_into_wall, pin_vel[:, 1] - dot_product * dir_out_y * blocking_factor, pin_vel[:, 1])
        
        pin_vel = torch.clamp(pin_vel, -250.0, 250.0)
        
        self.pin_pos[~self.pin_captured] += pin_vel[~self.pin_captured] * dt
        
        # === ЖЕСТКОЕ ВЫТАЛКИВАНИЕ ПРИ ГЛУБОКОМ ПРОНИКНОВЕНИИ (PENETRATION RESOLUTION) ===
        # Если нода все же оказалась внутри стены, принудительно выталкиваем ее координаты по нормали наружу
        pushed_mask = w_val > limit
        if pushed_mask.any():
            push_dist = (w_val - limit) * 80.0 * dt
            self.pin_pos[pushed_mask, 0] += dir_out_x[pushed_mask] * push_dist[pushed_mask]
            self.pin_pos[pushed_mask, 1] += dir_out_y[pushed_mask] * push_dist[pushed_mask]
        
        # Притягиваем к виртуальному порталу
        virtual_portal_pos = torch.stack([self.pin_pos[:, 0] - dx_p, self.pin_pos[:, 1] - dy_p], dim=1)
        self.pin_pos[self.pin_captured] = virtual_portal_pos[self.pin_captured] 

        self.player_density = self.solver.advect(self.player_density, self.u, self.v, dt * 0.4)
        self.density = self.solver.advect(self.density, self.u, self.v, dt * 0.4)

        gaussians = torch.exp(-(dx_shape**2 + dy_shape**2) / node_radius) * (1.0 - cap_mask_f.reshape(16, 1, 1))
        resting_player_density = torch.max(gaussians, dim=0, keepdim=True)[0].unsqueeze(0)

        if is_real_data:
            if eeg_c0_spectrum is not None and eeg_freqs is not None:
                F_bins = eeg_freqs.shape[0]
                
                # В разжатом состоянии расширяем радиус проекции частотных спектров на среду
                if compression < 0.0:
                    expand_ratio = -compression
                    dynamic_radius = (200.0 / (eeg_freqs + 2.0)) * (1.0 + expand_ratio * 3.0)
                else:
                    dynamic_radius = 200.0 / (eeg_freqs + 2.0) 

                is_active_1d = (~self.pin_captured).float()
                active_matrix = is_active_1d.unsqueeze(1) * is_active_1d.unsqueeze(0)
                c0_gpu = eeg_c0_spectrum[:16, :16, :] * active_matrix.unsqueeze(-1) 
                
                raw_diffs = self.pin_pos.unsqueeze(0) - self.pin_pos.unsqueeze(1)
                directions = raw_diffs / (torch.norm(raw_diffs, dim=2, keepdim=True) + 1e-5)
                
                mid_gx = torch.remainder(((pin_gx.unsqueeze(1) + pin_gx.unsqueeze(0)) * 0.5).reshape(256), self.res)
                mid_gy = torch.remainder(((pin_gy.unsqueeze(1) + pin_gy.unsqueeze(0)) * 0.5).reshape(256), self.res)
                
                dx_grid = torch.remainder(self.x_indices.reshape(self.res * self.res, 1) - mid_gx.unsqueeze(0) + self.res/2, self.res) - self.res/2
                dy_grid = torch.remainder(self.y_indices.reshape(self.res * self.res, 1) - mid_gy.unsqueeze(0) + self.res/2, self.res) - self.res/2
                D = dx_grid**2 + dy_grid**2 
                
                c0_flat = c0_gpu.reshape(256, F_bins) 
                dir_x = directions[:, :, 0].reshape(256, 1) 
                dir_y = directions[:, :, 1].reshape(256, 1) 
                
                force_x = dir_x * c0_flat 
                force_y = dir_y * c0_flat 
                
                total_force_x = torch.zeros((self.res * self.res), device=self.device)
                total_force_y = torch.zeros((self.res * self.res), device=self.device)
                
                for f in range(F_bins):
                    footprint_f = torch.exp(-D / dynamic_radius[f])
                    total_force_x += torch.matmul(footprint_f, force_x[:, f])
                    total_force_y += torch.matmul(footprint_f, force_y[:, f])
                
                total_force_x = (total_force_x / max(1, F_bins)).reshape(self.res, self.res)
                total_force_y = (total_force_y / max(1, F_bins)).reshape(self.res, self.res)

                # Использование настраиваемого eeg_force_scale
                self.u[0, 0] += self.player_density[0, 0] * total_force_x * self.cfg.get('eeg_force_scale', 150.0) * dt
                self.v[0, 0] += self.player_density[0, 0] * total_force_y * self.cfg.get('eeg_force_scale', 150.0) * dt
                
                force_mag = torch.sqrt(total_force_x**2 + total_force_y**2)
                injection_mask = self.player_density[0, 0] * (force_mag > 0.05).float()
                self.density[0, 0] += injection_mask * torch.clamp(force_mag / 5.0, 0.0, 1.0)
                self.density[0, 1] += injection_mask * torch.clamp(force_mag / 8.0, 0.0, 0.8)
                self.density[0, 2] += injection_mask * torch.clamp(force_mag / 20.0, 0.0, 0.2)
            
            else:
                # MANUAL CONTROL PHYSICS
                world_vx = eeg_vx * cos_p + eeg_vy * sin_p
                world_vy = -eeg_vx * sin_p + eeg_vy * cos_p
                
                if compression < 0.0:
                    # --- ВИХРЕВАЯ СЕНСОРНАЯ СЕТЬ (МНОГОТОЧЕЧНОЕ РАСПРЕДЕЛЕННОЕ УПРАВЛЕНИЕ) ---
                    expand_ratio = -compression
                    
                    for i in range(16):
                        if self.pin_captured[i]:
                            continue
                        
                        # Расчет полярных радиус-векторов относительно центра масс
                        dx_com = self.pin_pos[i, 0] - com[0]
                        dy_com = self.pin_pos[i, 1] - com[1]
                        dist_com = torch.sqrt(dx_com**2 + dy_com**2) + 1e-5
                        
                        # Тангенциальные векторы для вращательного движения нод вокруг центра
                        tangent_x = -dy_com / dist_com
                        tangent_y = dx_com / dist_com
                        
                        swirl_vx = tangent_x * eeg_tq * self.cfg.get('manual_force_scale', 600.0) * 0.7 * expand_ratio
                        swirl_vy = tangent_y * eeg_tq * self.cfg.get('manual_force_scale', 600.0) * 0.7 * expand_ratio
                        
                        # Центростремительное втягивание среды для захвата плотности
                        suction_vx = -dx_com / dist_com * self.cfg.get('manual_force_scale', 600.0) * 0.2 * expand_ratio
                        suction_vy = -dy_com / dist_com * self.cfg.get('manual_force_scale', 600.0) * 0.2 * expand_ratio
                        
                        node_gx_rem = pin_gx[i]
                        node_gy_rem = pin_gy[i]
                        
                        dx_grid_n = torch.remainder(self.x_indices - node_gx_rem + self.res/2, self.res) - self.res/2
                        dy_grid_n = torch.remainder(self.y_indices - node_gy_rem + self.res/2, self.res) - self.res/2
                        node_local_footprint = torch.exp(-(dx_grid_n**2 + dy_grid_n**2) / 8.0)
                        
                        # Индивидуальное ускорение на каждую точку гигантского невода
                        total_node_vx = swirl_vx + suction_vx + (world_vx * self.cfg.get('manual_force_scale', 600.0) * 0.4 * expand_ratio)
                        total_node_vy = swirl_vy + suction_vy + (world_vy * self.cfg.get('manual_force_scale', 600.0) * 0.4 * expand_ratio)
                        
                        self.u[0, 0] += node_local_footprint * total_node_vx * dt
                        self.v[0, 0] += node_local_footprint * total_node_vy * dt
                        
                        # Рисуем вихревые следы светящейся плотностью
                        self.density[0, 0] += node_local_footprint * 0.15 * expand_ratio
                        self.density[0, 2] += node_local_footprint * 0.3 * expand_ratio
                    
                    # Мягкий компенсирующий импульс в центре масс для сохранения инерции структуры
                    mid_gx = torch.remainder((com[0] / self.WIDTH) * self.res, self.res)
                    mid_gy = torch.remainder((com[1] / self.HEIGHT) * self.res, self.res)
                    dx_grid = torch.remainder(self.x_indices - mid_gx + self.res/2, self.res) - self.res/2
                    dy_grid = torch.remainder(self.y_indices - mid_gy + self.res/2, self.res) - self.res/2
                    footprint = torch.exp(-(dx_grid**2 + dy_grid**2) / 12.0)
                    self.u[0, 0] += footprint * world_vx * self.cfg.get('manual_force_scale', 600.0) * (1.0 - expand_ratio) * dt
                    self.v[0, 0] += footprint * world_vy * self.cfg.get('manual_force_scale', 600.0) * (1.0 - expand_ratio) * dt
                else:
                    # СТАНДАРТНОЕ СОСТОЯНИЕ И РЕЖИМ ЖЕСТКОГО НЕЙРОГЕЙМПАДА
                    mid_gx = torch.remainder((com[0] / self.WIDTH) * self.res, self.res)
                    mid_gy = torch.remainder((com[1] / self.HEIGHT) * self.res, self.res)
                    
                    dx_grid = torch.remainder(self.x_indices - mid_gx + self.res/2, self.res) - self.res/2
                    dy_grid = torch.remainder(self.y_indices - mid_gy + self.res/2, self.res) - self.res/2
                    
                    manual_radius = 12.0
                    footprint = torch.exp(-(dx_grid**2 + dy_grid**2) / manual_radius)
                    
                    self.u[0, 0] += footprint * world_vx * self.cfg.get('manual_force_scale', 600.0) * dt
                    self.v[0, 0] += footprint * world_vy * self.cfg.get('manual_force_scale', 600.0) * dt
                    
                    force_mag = math.sqrt(world_vx**2 + world_vy**2)
                    if force_mag > 0.1:
                        back_x = -world_vx / (force_mag + 1e-5)
                        back_y = -world_vy / (force_mag + 1e-5)
                        
                        mid_gx_back = torch.remainder(mid_gx + back_x * 4.0, self.res)
                        mid_gy_back = torch.remainder(mid_gy + back_y * 4.0, self.res)
                        
                        dx_grid_back = torch.remainder(self.x_indices - mid_gx_back + self.res/2, self.res) - self.res/2
                        dy_grid_back = torch.remainder(self.y_indices - mid_gy_back + self.res/2, self.res) - self.res/2
                        footprint_back = torch.exp(-(dx_grid_back**2 + dy_grid_back**2) / manual_radius)
                        
                        self.density[0, 0] += footprint_back * 0.4
                        self.density[0, 1] += footprint_back * 0.6
                
                # При сжатии делаем поворот в ручном режиме более отзывчивым
                if compression > 0.0:
                    turn_mult = 3.0 + blend * 2.0
                    self.player_angle += eeg_tq * turn_mult * dt
                else:
                    self.player_angle += eeg_tq * 3.0 * dt
                self.player_angle = (self.player_angle + math.pi) % (2.0 * math.pi) - math.pi

        goal_gx = (self.portal_pos[0] / self.WIDTH) * self.res
        goal_gy = (self.portal_pos[1] / self.HEIGHT) * self.res
        
        # Тороидальная воронка портала
        dx_portal = torch.remainder(self.x_indices - goal_gx + self.res/2, self.res) - self.res/2
        dy_portal = torch.remainder(self.y_indices - goal_gy + self.res/2, self.res) - self.res/2
        dist_portal = torch.sqrt(dx_portal**2 + dy_portal**2) + 1e-5
        
        core_mask = torch.exp(-(dist_portal**2) / ((self.res / self.maze.dim * 0.4)**2))
        self.density[0, 0] -= core_mask * 3.0 * dt
        self.density[0, 1] -= core_mask * 3.0 * dt
        self.density[0, 2] += core_mask * 2.5 * dt 
        self.density = torch.clamp(self.density, 0.0, 1.0)

        self.u = self.solver.advect(self.u, self.u, self.v, dt)
        self.v = self.solver.advect(self.v, self.u, self.v, dt)
        self.density = self.solver.advect(self.density, self.u, self.v, dt)
        
        erosion = torch.sqrt(self.u**2 + self.v**2) * 0.15 * dt
        self.wall_density = torch.clamp(self.wall_density - erosion, 0.0, 1.0)
        self.wall_density += (self.orig_obstacles - self.wall_density) * 0.15 * dt
        self.wall_density = torch.clamp(self.wall_density, 0.0, 1.0)
        
        self.u, self.v = self.solver.project(self.u, self.v, self.wall_density)

        if torch.sum(self.player_density) < 1e-4:
            self.player_density.copy_(resting_player_density)
            
        self.player_density = self.solver.advect(self.player_density, self.u, self.v, dt * 1.5)
        self.player_density += (resting_player_density - self.player_density) * 4.5 * dt
        
        hard_walls = (self.wall_density > 0.1).float()
        self.player_density = self.player_density * (1.0 - hard_walls)
        self.player_density = torch.clamp(self.player_density, 0.0, 1.0)

        self.player_pos[0] = com[0]
        self.player_pos[1] = com[1]

        u_pad = F.pad(self.u, (1, 1, 1, 1), mode='circular')
        v_pad = F.pad(self.v, (1, 1, 1, 1), mode='circular')
        vorticity = 0.5 * (v_pad[:, :, 1:-1, 2:] - v_pad[:, :, 1:-1, :-2]) - 0.5 * (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1])
        
        d_sum = torch.sum(self.player_density) + 1e-6
        raw_vorticity = (torch.sum(vorticity * self.player_density) / d_sum).item()
        
        self.smooth_vorticity = self.smooth_vorticity * 0.90 + raw_vorticity * 0.10
        
        if is_real_data:
            turn_speed = self.smooth_vorticity * self.cfg['vorticity_sensitivity'] * 1.2
            turn_speed = max(-1.2, min(1.2, turn_speed))
            
            self.player_angle += turn_speed * dt
            self.player_angle = (self.player_angle + math.pi) % (2.0 * math.pi) - math.pi

        if self.pin_captured.sum().item() == 16:
            self.reset_world()
