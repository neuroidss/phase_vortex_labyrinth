# vortex_renderer.py
import pygame
import torch
import torch.nn.functional as F
import numpy as np
import math

class VortexRenderer:
    def __init__(self, width, height, zoom_factor=1.35):
        self.WIDTH = width
        self.HEIGHT = height
        self.ZOOM = zoom_factor

    def render_field(self, arena):
        theta = -arena.player_angle
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        
        px_norm = (arena.player_pos[0] / self.WIDTH) * 2.0 - 1.0
        py_norm = (arena.player_pos[1] / self.HEIGHT) * 2.0 - 1.0
        
        M = torch.tensor([[
            [cos_t * self.ZOOM, -sin_t * self.ZOOM, px_norm],
            [sin_t * self.ZOOM,  cos_t * self.ZOOM, py_norm]
        ]], dtype=torch.float32, device=arena.device)
        
        grid = F.affine_grid(M, size=(1, 3, arena.res, arena.res), align_corners=True)
        
        # Считываем опцию "скрыть повторяющиеся лабиринты" из конфигурации физики
        hide_tiled = arena.cfg.get('hide_tiled_labyrinths', True)
        if not hide_tiled:
            # ТОРОИДАЛЬНОЕ ЗАВЕРТЫВАНИЕ: тайлим во все стороны
            grid = torch.remainder(grid + 1.0, 2.0) - 1.0
        
        cam_density = F.grid_sample(arena.density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        cam_walls = F.grid_sample(arena.wall_density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        cam_player = F.grid_sample(arena.player_density, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        vis = torch.clamp(cam_density[0].permute(1, 2, 0), 0.0, 1.0)
        vis = torch.where(vis == 0.0, torch.tensor([12.0/255.0]*3, device=arena.device), vis)
        
        wall_val = cam_walls[0, 0].unsqueeze(-1)
        wall_color = torch.tensor([0.9, 0.15, 0.6], device=arena.device).view(1, 1, 3)
        vis = vis * (1.0 - wall_val * 0.94) + wall_color * wall_val * 0.85
        
        player_val = cam_player[0, 0].unsqueeze(-1)
        jelly_color = torch.tensor([0.0, 0.45, 0.65], device=arena.device).view(1, 1, 3)
        membrane_color = torch.tensor([0.2, 1.0, 0.95], device=arena.device).view(1, 1, 3)
        membrane_mask = torch.clamp(1.0 - torch.abs(player_val - 0.18) / 0.08, 0.0, 1.0) ** 3.0
        
        vis = vis * (1.0 - player_val * 0.6) + jelly_color * player_val * 0.6
        vis = vis * (1.0 - membrane_mask * 0.8) + membrane_color * membrane_mask * 0.95
        
        cam_orig = F.grid_sample(arena.orig_obstacles, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        orig_val = cam_orig[0, 0].unsqueeze(-1)
        vis = torch.where(orig_val > 0.5, vis * 0.45 + torch.tensor([0.22, 0.05, 0.5], device=arena.device) * 0.55, vis)
        
        # === СВЕРХБЫСТРЫЙ РЕНДЕРИНГ: Аппаратное масштабирование сетки до размеров экрана на GPU ===
        vis_transposed = vis.permute(2, 0, 1).unsqueeze(0)  # [1, 3, res, res]
        scaled_vis = F.interpolate(vis_transposed, size=(self.WIDTH, self.HEIGHT), mode='bilinear', align_corners=True)
        rgb = (scaled_vis[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        return surf

    def draw_tension_lines(self, surface, arena):
        u_cpu, v_cpu = arena.u[0, 0].cpu().numpy(), arena.v[0, 0].cpu().numpy()
        theta, cos_t, sin_t = -arena.player_angle, math.cos(-arena.player_angle), math.sin(-arena.player_angle)
        step = arena.res // 24
        
        for i in range(1, 23):
            for j in range(1, 23):
                gy, gx = i * step, j * step
                wx, wy = (gx / float(arena.res)) * self.WIDTH, (gy / float(arena.res)) * self.HEIGHT
                
                # Короткий путь на Торе
                dx = (wx - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
                dy = (wy - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
                
                sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
                sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
                
                if 0 <= sx <= self.WIDTH and 0 <= sy <= self.HEIGHT:
                    vx_w, vy_w = float(u_cpu[gy, gx]), float(v_cpu[gy, gx])
                    vx_cam = (vx_w * cos_t + vy_w * sin_t) / self.ZOOM
                    vy_cam = (-vx_w * sin_t + vy_w * cos_t) / self.ZOOM
                    
                    speed = math.hypot(vx_cam, vy_cam)
                    if speed > 0.5:
                        max_len = 15.0
                        draw_vx = (vx_cam / speed) * max_len if speed > max_len else vx_cam
                        draw_vy = (vy_cam / speed) * max_len if speed > max_len else vy_cam
                        
                        ex, ey = sx + draw_vx * 1.5, sy + draw_vy * 1.5
                        if math.isfinite(ex) and math.isfinite(ey):
                            col_f = min(1.0, speed / 40.0)
                            pygame.draw.line(surface, (0, int(150 + col_f * 105), int(255 - col_f * 100)), (int(sx), int(sy)), (int(ex), int(ey)), 1)

    def draw_coherence_bridges(self, surface, arena):
        """Рисует упругие перемычки (когерентности) между всеми ядрами слайма"""
        c0_matrix = arena.eeg_c0_matrix.cpu().numpy()
        theta, cos_t, sin_t = -arena.player_angle, math.cos(-arena.player_angle), math.sin(-arena.player_angle)
        
        # Получаем экранные координаты всех 16 нод слайма
        screen_coords = []
        for i in range(16):
            px, py = arena.pin_pos[i, 0].item(), arena.pin_pos[i, 1].item()
            dx = (px - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
            dy = (py - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
            
            sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
            screen_coords.append((int(sx), int(sy)))
            
        # Рисуем линии связей
        max_val = c0_matrix.max() if c0_matrix.max() > 1e-5 else 1.0
        for i in range(16):
            for j in range(i + 1, 16):
                val = float(c0_matrix[i, j])
                if val > 0.05:
                    # Плавный цвет от бирюзового к ярко-розовому в зависимости от силы когерентности связи
                    ratio = val / max_val
                    col = (int(50 + 205 * ratio), int(255 - 155 * ratio), 255)
                    thickness = max(1, int(ratio * 4.5))
                    pygame.draw.line(surface, col, screen_coords[i], screen_coords[j], thickness)

    def draw_electrode_sensors(self, surface, arena):
        # Рисуем упругие перемычки (когерентности) между всеми ядрами
        if hasattr(arena, 'eeg_c0_matrix') and arena.eeg_c0_matrix.sum().item() > 0.05:
            self.draw_coherence_bridges(surface, arena)

        theta, cos_t, sin_t = -arena.player_angle, math.cos(-arena.player_angle), math.sin(-arena.player_angle)
        
        for i in range(16):
            is_cap = arena.pin_captured[i]
            col = (255, 0, 255) if is_cap else (0, 255, 255)
            
            px, py = arena.pin_pos[i, 0].item(), arena.pin_pos[i, 1].item()
            dx = (px - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
            dy = (py - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
            
            sx = self.WIDTH / 2.0 + (dx * cos_t + dy * sin_t) / self.ZOOM
            sy = self.HEIGHT / 2.0 + (-dx * sin_t + dy * cos_t) / self.ZOOM
            
            pygame.draw.circle(surface, (*col, 100), (int(sx), int(sy)), 8, 1)
            pygame.draw.circle(surface, col, (int(sx), int(sy)), 2)

    def draw_ui(self, surface, arena):
        cell_w = (self.WIDTH * 0.8) / arena.maze.dim
        gx = (self.WIDTH * 0.1) + (arena.goal_cell[0] + 0.5) * cell_w
        gy = (self.HEIGHT * 0.1) + (arena.goal_cell[1] + 0.5) * cell_w
        
        dx = (gx - arena.player_pos[0].item() + self.WIDTH/2) % self.WIDTH - self.WIDTH/2
        dy = (gy - arena.player_pos[1].item() + self.HEIGHT/2) % self.HEIGHT - self.HEIGHT/2
        
        theta = -arena.player_angle
        sx = self.WIDTH / 2.0 + (dx * math.cos(theta) + dy * math.sin(theta)) / self.ZOOM
        sy = self.HEIGHT / 2.0 + (-dx * math.sin(theta) + dy * math.cos(theta)) / self.ZOOM
        
        if 0 <= sx <= self.WIDTH and 0 <= sy <= self.HEIGHT:
            pygame.draw.circle(surface, (0, 255, 100), (int(sx), int(sy)), int(cell_w * 0.25 / self.ZOOM), 1)

        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 14, 1)
        pygame.draw.circle(surface, (0, 255, 255), (cx, cy), 4)
