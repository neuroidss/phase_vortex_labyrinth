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
from vortex_telemetry import update_rune_zones


class PhaseVortexArena:
    def __init__(self, device, width, height, res, seed=202607):
        self.device = device
        self.WIDTH, self.HEIGHT = width, height
        self.res = res
        self.current_seed = seed
        self.solver = FluidSolver(res, device)
        
        # === ЦЕНТРАЛИЗОВАННЫЙ СЛОВАРЬ НАСТРОЕК СИМУЛЯЦИИ ===
        self.cfg = {
            # РЕЖИМЫ ЭВОЛЮЦИИ ИГРОКА 
            'bci_mode': '120_jets', 
            
            # АБСОЛЮТНЫЕ ОСИ (Идеальная управляемость в любом угле поворота)
            'coherence_relative_to_physical': True,  
            
            'sticky_walls': False,   

            'vorticity_sensitivity': 0.15, 
            'fluid_damping': 0.80,
            
            # Базовые лимиты проникновения
            'inner_wall_penetration_limit': 0.04,   
            'outer_wall_repulsion_scale': 150000.0, 
            'outer_wall_penetration_limit': 0.001,  
            
            'hide_tiled_labyrinths': True,    
            'show_debug_window': True,       
            
            'torque_sensitivity_multiplier': 0.0008, 
            
            # === ПАРАМЕТРЫ ЧИСТОЙ ГИДРОДИНАМИКИ КАПЛИ ===
            'base_scale': 1.25,              
            'compress_scale_mult': 0.45,     
            'expand_scale_mult': 1.25,       
            
            'base_node_radius': 8.0,         
            'compress_radius_sub': 4.0,      
            # Уменьшен разброс в разжатом виде, чтобы сопла не распылялись слишком сильно
            'expand_radius_add': 4.0,        
            
            'fluid_cohesion_force': 45.0,    
            'fluid_cohesion_gravity': 120.0, 
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
        
        self.pair_i, self.pair_j = torch.triu_indices(16, 16, offset=1, device=device)
        
        dx_base = self.pin_x.unsqueeze(0) - self.pin_x.unsqueeze(1)
        dy_base = self.pin_y.unsqueeze(0) - self.pin_y.unsqueeze(1)
        self.base_dist = torch.sqrt(dx_base**2 + dy_base**2)
        
        self.pin_pos = torch.zeros((16, 2), dtype=torch.float32, device=device)
        self.pin_captured = torch.zeros(16, dtype=torch.bool, device=device)
        self.edge_intact = torch.ones(16, dtype=torch.bool, device=device)
        
        self.player_pos = torch.tensor([width/2, height/2], dtype=torch.float32, device=device)
        self.portal_pos = torch.zeros(2, dtype=torch.float32, device=device)
        self.player_angle = 0.0
        self.player_angular_vel = 0.0
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
        self.rune_zones = []

    def step(self, dt, time_sec, eeg_c0_spectrum, eeg_vx, eeg_vy, eeg_tq, is_real_data, compression, scale_factor, eeg_freqs=None):
        if torch.is_tensor(eeg_vx): eeg_vx = eeg_vx.item()
        if torch.is_tensor(eeg_vy): eeg_vy = eeg_vy.item()
        if torch.is_tensor(eeg_tq): eeg_tq = eeg_tq.item()

        blend = max(-1.0, min(1.0, compression))
        
        if blend < 0.0:
            scale = self.cfg['base_scale'] - blend * self.cfg['expand_scale_mult']
            node_radius = self.cfg['base_node_radius'] - blend * self.cfg['expand_radius_add']
        else:
            scale = self.cfg['base_scale'] - blend * self.cfg['compress_scale_mult']
            node_radius = self.cfg['base_node_radius'] - blend * self.cfg['compress_radius_sub']

        damping_factor = 0.96 - abs(blend) * 0.02
        self.u = torch.nan_to_num(self.u, nan=0.0) * damping_factor
        self.v = torch.nan_to_num(self.v, nan=0.0) * damping_factor
        self.density = torch.nan_to_num(self.density, nan=0.0) * 0.985
        
        self.u = torch.clamp(self.u, -55.0, 55.0)
        self.v = torch.clamp(self.v, -55.0, 55.0)
        
        erosion = torch.sqrt(self.u**2 + self.v**2) * 0.005 * dt
        self.inner_wall_density = torch.clamp(self.inner_obstacles - erosion, 0.0, 1.0)
        self.inner_wall_density += (self.inner_obstacles - self.inner_wall_density) * 1.8 * dt
        self.inner_wall_density = torch.clamp(self.inner_wall_density, 0.0, 1.0)
        self.wall_density = torch.clamp(self.inner_wall_density + self.outer_obstacles, 0.0, 1.0)

        if self.pin_captured.all():
            self.reset_world()
            return
            
        com = self.pin_pos.mean(dim=0)
        use_gamepad_math = (self.cfg['bci_mode'] == 'Neurogamepad') or not is_real_data or eeg_c0_spectrum is None

        # === ГИБРИДНАЯ ПРЕДСКАЗЫВАЮЩАЯ КАМЕРА ===
        actual_local_x = self.pin_pos[:, 0] - com[0]
        actual_local_y = self.pin_pos[:, 1] - com[1]
        
        cross_cov = torch.sum(self.pin_y * actual_local_x - self.pin_x * actual_local_y)
        dot_cov = torch.sum(self.pin_x * actual_local_x + self.pin_y * actual_local_y) + 1e-5
        self.debug_cross_cov, self.debug_dot_cov = cross_cov.item(), dot_cov.item()
        
        raw_angle = torch.atan2(cross_cov, dot_cov).item()
        angle_diff = (raw_angle - self.player_angle + math.pi) % (2 * math.pi) - math.pi
        
        self.player_angle += angle_diff * 0.40 
        self.player_angle -= eeg_tq * 3.5 * dt 

        cos_p, sin_p = math.cos(self.player_angle), math.sin(self.player_angle)
        
        ideal_x = self.pin_x * cos_p + self.pin_y * sin_p
        ideal_y = -self.pin_x * sin_p + self.pin_y * cos_p
        ideal_x_scaled = ideal_x * scale
        ideal_y_scaled = ideal_y * scale
        ideal_pos = com.unsqueeze(0) + torch.stack([ideal_x_scaled, ideal_y_scaled], dim=1)
        
        is_active_1d = (~self.pin_captured).float()
        active_matrix = is_active_1d.unsqueeze(1) * is_active_1d.unsqueeze(0)

        dist_local = torch.sqrt(actual_local_x**2 + actual_local_y**2) + 1e-5
        tangent_x = -actual_local_y / dist_local
        tangent_y = actual_local_x / dist_local

        # === ГЕНЕРАЦИЯ СИЛ В ЖИДКОСТЬ ===
        bci_force_grid_x = torch.zeros((self.res, self.res), device=self.device)
        bci_force_grid_y = torch.zeros((self.res, self.res), device=self.device)
        bci_mag = 0.0
        
        node_coherence = torch.zeros(16, device=self.device)

        if use_gamepad_math:
            forward_speed, strafe_speed = -eeg_vy, eeg_vx
            
            world_vx = -math.sin(self.player_angle) * forward_speed + math.cos(self.player_angle) * strafe_speed
            world_vy = -math.cos(self.player_angle) * forward_speed - math.sin(self.player_angle) * strafe_speed
            
            bci_mag = math.sqrt(world_vx**2 + world_vy**2)
            
            force_mult_gp = 400.0 * (1.0 + max(0.0, blend) * 2.0)
            node_bci_force_x = torch.full((16,), world_vx, device=self.device) * force_mult_gp
            node_bci_force_y = torch.full((16,), world_vy, device=self.device) * force_mult_gp
            
            torque_fluid_mult = 200.0 * (1.0 + max(0.0, blend) * 2.0)
            node_bci_force_x += tangent_x * eeg_tq * torque_fluid_mult
            node_bci_force_y += tangent_y * eeg_tq * torque_fluid_mult
            
            node_coherence = torch.full((16,), bci_mag * 0.5 + 0.1, device=self.device)
            self.eeg_c0_matrix.zero_()
            
            pin_gx = torch.remainder((self.pin_pos[:, 0] / self.WIDTH) * self.res, self.res)
            pin_gy = torch.remainder((self.pin_pos[:, 1] / self.HEIGHT) * self.res, self.res)
            dx_shape = torch.remainder(self.x_indices.unsqueeze(0) - pin_gx.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
            dy_shape = torch.remainder(self.y_indices.unsqueeze(0) - pin_gy.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
            node_influence = torch.exp(-(dx_shape**2 + dy_shape**2) / node_radius) * is_active_1d.reshape(16, 1, 1)
            node_influence_norm = node_influence / (torch.sum(node_influence, dim=(1, 2), keepdim=True) + 1e-8)
            
            bci_force_grid_x = torch.sum(node_influence_norm * node_bci_force_x.reshape(16, 1, 1), dim=0)
            bci_force_grid_y = torch.sum(node_influence_norm * node_bci_force_y.reshape(16, 1, 1), dim=0)

        elif self.cfg['bci_mode'] == '16_jets':
            c0_gpu = eeg_c0_spectrum[:16, :16, :] * active_matrix.unsqueeze(-1)
            c0_total = torch.mean(c0_gpu, dim=2) 
            self.eeg_c0_matrix.copy_(c0_total)
            
            dx_ideal = ideal_x.unsqueeze(1) - ideal_x.unsqueeze(0) if self.cfg['coherence_relative_to_physical'] else self.pin_x.unsqueeze(1) - self.pin_x.unsqueeze(0)
            dy_ideal = ideal_y.unsqueeze(1) - ideal_y.unsqueeze(0) if self.cfg['coherence_relative_to_physical'] else self.pin_y.unsqueeze(1) - self.pin_y.unsqueeze(0)
            
            force_multiplier = (40.0 - min(0.0, blend) * 15.0) * (1.0 + max(0.0, blend) * 1.5)
            node_bci_force_x = torch.sum(c0_total * dx_ideal, dim=1) * force_multiplier
            node_bci_force_y = torch.sum(c0_total * dy_ideal, dim=1) * force_multiplier
            
            node_bci_force_x += tangent_x * eeg_tq * 100.0
            node_bci_force_y += tangent_y * eeg_tq * 100.0
            
            bci_mag = torch.sqrt(torch.sum(node_bci_force_x)**2 + torch.sum(node_bci_force_y)**2).item() / 100.0
            node_coherence = torch.sum(torch.abs(c0_total), dim=1)
            
            pin_gx = torch.remainder((self.pin_pos[:, 0] / self.WIDTH) * self.res, self.res)
            pin_gy = torch.remainder((self.pin_pos[:, 1] / self.HEIGHT) * self.res, self.res)
            dx_shape = torch.remainder(self.x_indices.unsqueeze(0) - pin_gx.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
            dy_shape = torch.remainder(self.y_indices.unsqueeze(0) - pin_gy.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
            node_influence = torch.exp(-(dx_shape**2 + dy_shape**2) / node_radius) * is_active_1d.reshape(16, 1, 1)
            node_influence_norm = node_influence / (torch.sum(node_influence, dim=(1, 2), keepdim=True) + 1e-8)
            
            bci_force_grid_x = torch.sum(node_influence_norm * node_bci_force_x.reshape(16, 1, 1), dim=0)
            bci_force_grid_y = torch.sum(node_influence_norm * node_bci_force_y.reshape(16, 1, 1), dim=0)

        elif self.cfg['bci_mode'] == '120_jets':
            c0_gpu = eeg_c0_spectrum[:16, :16, :] * active_matrix.unsqueeze(-1)
            c0_total = torch.mean(c0_gpu, dim=2) 
            self.eeg_c0_matrix.copy_(c0_total)
            
            c0_120 = c0_total[self.pair_i, self.pair_j]
            
            mid_x = (self.pin_pos[self.pair_i, 0] + self.pin_pos[self.pair_j, 0]) / 2.0
            mid_y = (self.pin_pos[self.pair_i, 1] + self.pin_pos[self.pair_j, 1]) / 2.0
            
            dx_ideal = ideal_x[self.pair_i] - ideal_x[self.pair_j] if self.cfg['coherence_relative_to_physical'] else self.pin_x[self.pair_i] - self.pin_x[self.pair_j]
            dy_ideal = ideal_y[self.pair_i] - ideal_y[self.pair_j] if self.cfg['coherence_relative_to_physical'] else self.pin_y[self.pair_i] - self.pin_y[self.pair_j]
            
            # Динамическое усиление тяги 120 вихрей в нормальном и разжатом режимах
            # Это дает силу вырваться из липких углов!
            force_multiplier = 16.0 - min(0.0, blend) * 8.0 
            jet_force_x = c0_120 * dx_ideal * force_multiplier
            jet_force_y = c0_120 * dy_ideal * force_multiplier
            
            actual_mid_x, actual_mid_y = mid_x - com[0], mid_y - com[1]
            dist_mid = torch.sqrt(actual_mid_x**2 + actual_mid_y**2) + 1e-5
            jet_force_x += (-actual_mid_y / dist_mid) * eeg_tq * 30.0
            jet_force_y += (actual_mid_x / dist_mid) * eeg_tq * 30.0
            
            bci_mag = torch.sqrt(torch.sum(jet_force_x)**2 + torch.sum(jet_force_y)**2).item() / 100.0
            node_coherence = torch.sum(torch.abs(c0_total), dim=1)
            
            pin_gx_120 = torch.remainder((mid_x / self.WIDTH) * self.res, self.res)
            pin_gy_120 = torch.remainder((mid_y / self.HEIGHT) * self.res, self.res)
            dx_shape_120 = torch.remainder(self.x_indices.unsqueeze(0) - pin_gx_120.reshape(120, 1, 1) + self.res/2, self.res) - self.res/2
            dy_shape_120 = torch.remainder(self.y_indices.unsqueeze(0) - pin_gy_120.reshape(120, 1, 1) + self.res/2, self.res) - self.res/2
            
            is_active_120 = (~self.pin_captured[self.pair_i]) & (~self.pin_captured[self.pair_j])
            node_influence_120 = torch.exp(-(dx_shape_120**2 + dy_shape_120**2) / (node_radius * 0.6)) * is_active_120.float().reshape(120, 1, 1)
            node_inf_120_norm = node_influence_120 / (torch.sum(node_influence_120, dim=(1, 2), keepdim=True) + 1e-8)
            
            bci_force_grid_x = torch.sum(node_inf_120_norm * jet_force_x.reshape(120, 1, 1), dim=0)
            bci_force_grid_y = torch.sum(node_inf_120_norm * jet_force_y.reshape(120, 1, 1), dim=0)

        # Семплирование скоростей для кинематики нод
        pin_uv_raw = (self.pin_pos / self.screen_size) * 2.0 - 1.0
        pin_uv = torch.remainder(pin_uv_raw + 1.0, 2.0) - 1.0
        sampled_u = F.grid_sample(self.u, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
        sampled_v = F.grid_sample(self.v, pin_uv.view(1, 1, 16, 2), mode='bilinear', align_corners=True).squeeze()
        fluid_vel = torch.stack([sampled_u, sampled_v], dim=1) * 80.0 

        # === КОНТИНУУМ ЛИПКОСТИ (Магнитная подушка стен) ===
        # blend = -1 -> 1500 (Слабая подушка, слайм вязнет в стене)
        # blend =  0 -> 3500 (Норма, удерживает сопла в "живой воде")
        # blend = +1 -> 5500 (Бронебойная подушка, мгновенный отскок)
        wall_repulsion = 3500.0 + blend * 2000.0
        
        inner_smooth = F.avg_pool2d(self.inner_wall_density, kernel_size=5, stride=1, padding=2)
        outer_smooth = F.avg_pool2d(self.outer_obstacles, kernel_size=5, stride=1, padding=2)
        w_inner_pad = F.pad(inner_smooth, (1, 1, 1, 1), mode='circular')
        grad_x_in = 0.5 * (w_inner_pad[:, :, 1:-1, 2:] - w_inner_pad[:, :, 1:-1, :-2])
        grad_y_in = 0.5 * (w_inner_pad[:, :, 2:, 1:-1] - w_inner_pad[:, :, :-2, 1:-1])
        w_gx_in = F.grid_sample(grad_x_in, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_gy_in = F.grid_sample(grad_y_in, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        
        w_outer_pad = F.pad(outer_smooth, (1, 1, 1, 1), mode='circular')
        grad_x_out = 0.5 * (w_outer_pad[:, :, 1:-1, 2:] - w_outer_pad[:, :, 1:-1, :-2])
        grad_y_out = 0.5 * (w_outer_pad[:, :, 2:, 1:-1] - w_outer_pad[:, :, :-2, 1:-1])
        w_gx_out = F.grid_sample(grad_x_out, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_gy_out = F.grid_sample(grad_y_out, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        
        w_gx, w_gy = w_gx_in + w_gx_out, w_gy_in + w_gy_out
        grad_norm = torch.sqrt(w_gx**2 + w_gy**2) + 1e-5
        dir_out_x, dir_out_y = -w_gx / grad_norm, -w_gy / grad_norm
        
        w_val_smooth_in = F.grid_sample(inner_smooth, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_val_smooth_out = F.grid_sample(outer_smooth, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        w_val_sharp = F.grid_sample(self.wall_density, pin_uv.view(1, 1, 16, 2), align_corners=True).squeeze()
        
        f_wall_in = torch.stack([dir_out_x * w_val_smooth_in * wall_repulsion, 
                                 dir_out_y * w_val_smooth_in * wall_repulsion], dim=1)
        f_wall_out = torch.stack([dir_out_x * w_val_smooth_out * self.cfg['outer_wall_repulsion_scale'], 
                                  dir_out_y * w_val_smooth_out * self.cfg['outer_wall_repulsion_scale']], dim=1)
        f_wall = f_wall_in + f_wall_out
        
        dx_p = torch.remainder(self.pin_pos[:, 0] - self.portal_pos[0] + self.WIDTH/2, self.WIDTH) - self.WIDTH/2
        dy_p = torch.remainder(self.pin_pos[:, 1] - self.portal_pos[1] + self.HEIGHT/2, self.HEIGHT) - self.HEIGHT/2
        self.pin_captured = self.pin_captured | ((torch.sqrt(dx_p**2 + dy_p**2)) < self.cell_w * 0.35)
        
        # Силы формы
        if blend >= 0.0: shape_pull = 15.0 + blend * 35.0
        else: shape_pull = 15.0 + blend * 10.0 
            
        f_restore = (ideal_pos - self.pin_pos) * shape_pull
        f_spring, self.edge_intact = update_neighbor_springs(self.pin_pos, ideal_pos, self.edge_intact, blend, self.device)

        # Когезия плотности
        rho_pad = F.pad(self.player_density, (1, 1, 1, 1), mode='circular')
        grad_x_rho = 0.5 * (rho_pad[:, :, 1:-1, 2:] - rho_pad[:, :, 1:-1, :-2])
        grad_y_rho = 0.5 * (rho_pad[:, :, 2:, 1:-1] - rho_pad[:, :, :-2, 1:-1])
        cohesion_coeff = self.cfg['fluid_cohesion_force'] + max(0.0, blend) * 180.0
        f_cohesion_x = grad_x_rho[0, 0] * cohesion_coeff
        f_cohesion_y = grad_y_rho[0, 0] * cohesion_coeff

        com_grid_x = (self.player_pos[0] / self.WIDTH) * 2.0 - 1.0
        com_grid_y = (self.player_pos[1] / self.HEIGHT) * 2.0 - 1.0
        dx_com = torch.remainder((com_grid_x - self.solver.grid_x) + 1.0, 2.0) - 1.0
        dy_com = torch.remainder((com_grid_y - self.solver.grid_y) + 1.0, 2.0) - 1.0
        gravity_coeff = self.cfg['fluid_cohesion_gravity'] * (1.0 + max(0.0, blend) * 1.5)
        f_gravity_x = dx_com * self.player_density[0, 0] * gravity_coeff
        f_gravity_y = dy_com * self.player_density[0, 0] * gravity_coeff

        # Инжекция в жидкость
        self.u[0, 0] += (bci_force_grid_x * 1.5 + f_cohesion_x + f_gravity_x) * dt
        self.v[0, 0] += (bci_force_grid_y * 1.5 + f_cohesion_y + f_gravity_y) * dt

        # === 4. КИНЕМАТИКА ===
        slip_factor = 0.85 + blend * 0.15
        slip_factor = max(0.65, min(1.0, slip_factor))

        pin_vel = fluid_vel * slip_factor + f_wall + f_spring * 0.15 + f_restore

        # Вращение
        pin_vel[:, 0] += tangent_x * eeg_tq * 100.0
        pin_vel[:, 1] += tangent_y * eeg_tq * 100.0

        dot_inner = pin_vel[:, 0] * dir_out_x + pin_vel[:, 1] * dir_out_y
        limit_inner = self.cfg['inner_wall_penetration_limit']
        blocking_inner = torch.clamp(w_val_sharp / limit_inner, 0.0, 1.0)
        moving_into_inner = dot_inner < 0
        
        # Гашение перпендикулярной составляющей (Идеальное скольжение)
        pin_vel[:, 0] -= torch.where(moving_into_inner, dot_inner * dir_out_x * blocking_inner, torch.zeros_like(pin_vel[:, 0]))
        pin_vel[:, 1] -= torch.where(moving_into_inner, dot_inner * dir_out_y * blocking_inner, torch.zeros_like(pin_vel[:, 1]))
        
        pin_vel = torch.clamp(pin_vel, -180.0, 180.0)
        self.pin_pos[~self.pin_captured] += pin_vel[~self.pin_captured] * dt
        
        new_com = self.pin_pos.mean(dim=0)
        ideal_pos_new = new_com.unsqueeze(0) + torch.stack([ideal_x_scaled, ideal_y_scaled], dim=1)
        self.pin_pos = apply_cohesion_constraint(self.pin_pos, ideal_pos_new, self.pin_captured, scale, blend)
        
        # === 5. ПОСТ-ШАГОВОЕ ВЫТАЛКИВАНИЕ (Континуум Липкости) ===
        pin_uv_raw_post = (self.pin_pos / self.screen_size) * 2.0 - 1.0
        pin_uv_post = torch.remainder(pin_uv_raw_post + 1.0, 2.0) - 1.0
        w_val_sharp_post = F.grid_sample(self.wall_density, pin_uv_post.view(1, 1, 16, 2), align_corners=True).squeeze()
        
        # Континуум кинематического выталкивания из углов:
        # blend = -1 -> 4.0 (Очень мягко, слайм втекает в углы и ползет по ним)
        # blend =  0 -> 12.0 (Сбалансировано, выталкивает из "мертвой воды")
        # blend = +1 -> 20.0 (Жесткий отскок)
        push_mult = 12.0 + blend * 8.0
        
        pushed_wall = w_val_sharp_post > limit_inner
        if pushed_wall.any():
            push_dist = (w_val_sharp_post - limit_inner) * push_mult
            self.pin_pos[pushed_wall, 0] += dir_out_x[pushed_wall] * push_dist[pushed_wall]
            self.pin_pos[pushed_wall, 1] += dir_out_y[pushed_wall] * push_dist[pushed_wall]
            
        virtual_portal_pos = torch.stack([self.pin_pos[:, 0] - dx_p, self.pin_pos[:, 1] - dy_p], dim=1)
        self.pin_pos[self.pin_captured] = virtual_portal_pos[self.pin_captured] 

        # === 6. ОБНОВЛЕНИЕ ЖИДКОСТИ И РЕНДЕР ===
        self.player_density = self.solver.advect(self.player_density, self.u, self.v, dt * 0.4)
        self.density = self.solver.advect(self.density, self.u, self.v, dt * 0.4)

        if self.cfg['bci_mode'] == '120_jets' and not use_gamepad_math:
            coherence_120 = torch.abs(c0_total[self.pair_i, self.pair_j])
            smoke_grid = torch.sum(node_influence_120 * coherence_120.reshape(120, 1, 1), dim=0)
        else:
            pin_gx = torch.remainder((self.pin_pos[:, 0] / self.WIDTH) * self.res, self.res)
            pin_gy = torch.remainder((self.pin_pos[:, 1] / self.HEIGHT) * self.res, self.res)
            dx_shape = torch.remainder(self.x_indices.unsqueeze(0) - pin_gx.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
            dy_shape = torch.remainder(self.y_indices.unsqueeze(0) - pin_gy.reshape(16, 1, 1) + self.res/2, self.res) - self.res/2
            node_influence = torch.exp(-(dx_shape**2 + dy_shape**2) / node_radius) * is_active_1d.reshape(16, 1, 1)
            smoke_grid = torch.sum(node_influence * node_coherence.reshape(16, 1, 1), dim=0)

        injection_mask = smoke_grid * (smoke_grid > 0.05).float()
        self.density[0, 0] += injection_mask * min(1.0, max(0.0, bci_mag / 2.0)) * dt * 45.0
        self.density[0, 1] += injection_mask * min(0.8, max(0.0, bci_mag / 3.0)) * dt * 45.0
        self.density[0, 2] += injection_mask * min(0.2, max(0.0, bci_mag / 10.0)) * dt * 45.0

        goal_gx = (self.portal_pos[0] / self.WIDTH) * self.res
        goal_gy = (self.portal_pos[1] / self.HEIGHT) * self.res
        dx_portal = torch.remainder(self.x_indices - goal_gx + self.res/2, self.res) - self.res/2
        dy_portal = torch.remainder(self.y_indices - goal_gy + self.res/2, self.res) - self.res/2
        portal_radius = self.res / self.maze.dim * 0.4
        core_mask = torch.exp(-((dx_portal**2 + dy_portal**2)) / (portal_radius**2))
        
        self.density[0, 0] -= core_mask * 3.0 * dt
        self.density[0, 1] -= core_mask * 3.0 * dt
        self.density[0, 2] += core_mask * 2.5 * dt 
        self.density = torch.clamp(self.density, 0.0, 1.0)

        self.u = self.solver.advect(self.u, self.u, self.v, dt)
        self.v = self.solver.vvect(self.v, self.u, self.v, dt) if hasattr(self.solver, 'vvect') else self.solver.advect(self.v, self.u, self.v, dt)
        self.density = self.solver.advect(self.density, self.u, self.v, dt)
        
        self.u, self.v = self.solver.project(self.u, self.v, self.wall_density, None)

        self.density = self.density * (self.player_density * 0.95 + 0.05)
        
        if self.cfg['bci_mode'] == '120_jets' and not use_gamepad_math:
            resting_player_density = torch.max(node_influence_120, dim=0, keepdim=True)[0].unsqueeze(0)
        else:
            resting_player_density = torch.max(node_influence, dim=0, keepdim=True)[0].unsqueeze(0)
            
        if torch.sum(self.player_density) < 1e-4: self.player_density.copy_(resting_player_density)
            
        self.player_density = self.solver.advect(self.player_density, self.u, self.v, dt * 1.5)
        self.player_density += (resting_player_density - self.player_density) * 4.5 * dt
        
        self.player_density = torch.clamp(self.player_density, 0.0, 1.0)
        self.player_density = torch.clamp(self.player_density * (1.0 - (self.wall_density > 0.1).float()), 0.0, 1.0)

        self.player_pos[0], self.player_pos[1] = new_com[0], new_com[1]

        u_pad, v_pad = F.pad(self.u, (1, 1, 1, 1), mode='circular'), F.pad(self.v, (1, 1, 1, 1), mode='circular')
        vorticity = 0.5 * (v_pad[:, :, 1:-1, 2:] - v_pad[:, :, 1:-1, :-2]) - 0.5 * (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1])
        self.smooth_vorticity = self.smooth_vorticity * 0.90 + (torch.sum(vorticity * self.player_density) / (torch.sum(self.player_density) + 1e-6)).item() * 0.10

        update_rune_zones(
            self.rune_zones, self.player_pos, self.pin_pos,
            self.pin_captured, pin_vel, self.smooth_vorticity,
            self.player_angle, dt, self.device
        )

        if self.pin_captured.sum().item() == 16:
            self.reset_world()
